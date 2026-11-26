#!/usr/bin/env python3
"""Mutation run for the assertion checkers of tb/sva.

Injects one RTL defect at a time, runs the testbench that exercises it twice,
with the assertions and without them (make VERILATOR_ASSERT=), and prints which
assertion fails first and whether the testbench's own checks notice. This is
the evidence that the assertions can fire and what they add; it takes about
50 minutes and is not part of make ci.

With --cocotb it also runs the cocotb / pyuvm environment on every mutation
with the assertions compiled out, so only its scoreboard can notice (about
another 30 minutes; cocotb must be installed, tb/cocotb/requirements.txt).

With --uvm it runs the SystemVerilog UVM environment (make uvm, npu_full_test)
with the assertions compiled out, so only its scoreboard, register model and
sequence checks can notice (about 4 minutes per mutation; needs Verilator
5.052 or later as VERILATOR_UVM). --uvm-only skips the SVA / testbench runs.

Usage: mutation_run.py [--cocotb] [--uvm | --uvm-only] [substring of a mutation name]
The RTL tree must be clean (git); every file is restored after its run.
"""
import os
import re
import signal
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (name, file, text to replace, replacement, make target)
M = [
 ("axil B dropped without BREADY", "rtl/host/axil_slave.sv", "          if (s_axil_bready_i) state_q <= IDLE;", "          state_q <= IDLE;", "unit-axil_slave"),
 ("axil write accepted on AW alone", "rtl/host/axil_slave.sv", "(state_q == IDLE) && s_axil_awvalid_i && s_axil_wvalid_i;", "(state_q == IDLE) && s_axil_awvalid_i;", "unit-axil_slave"),
 ("axil RDATA not captured at ack", "rtl/host/axil_slave.sv", "            s_axil_rdata_o <= rd_data_i;", "            s_axil_rdata_o <= rd_data_i ^ 32'h1;", "unit-axil_slave"),
 ("dma ignores the 4 KB boundary", "rtl/host/npu_dma.sv", "    if (to_boundary_words_c < 20'(next_beats_c)) next_beats_c = 9'(to_boundary_words_c);", "", "integ-layers"),
 ("dma WLAST one beat late", "rtl/host/npu_dma.sv", "assign m_axi_wlast_o   = (beat_q == burst_beats_q - 8'd1);", "assign m_axi_wlast_o   = (beat_q == burst_beats_q);", "integ-layers"),
 ("dma AWLEN one too long", "rtl/host/npu_dma.sv", "assign m_axi_awlen_o   = burst_beats_q - 8'd1;", "assign m_axi_awlen_o   = burst_beats_q;", "integ-layers"),
 ("fifo write pointer sent in binary", "rtl/chip/async_fifo.sv", "      wr_gray_q <= bin2gray(wr_bin_next_c);", "      wr_gray_q <= wr_bin_next_c;", "unit-async_fifo"),
 ("fifo full compares without MSB inversion", "rtl/chip/async_fifo.sv", "{~rd_gray_in_wr[AW:AW-1], rd_gray_in_wr[AW-2:0]}", "{rd_gray_in_wr[AW], ~rd_gray_in_wr[AW-1], rd_gray_in_wr[AW-2:0]}", "unit-async_fifo"),
 ("requant valid one stage early", "rtl/compute/requant_unit.sv", "      valid_o <= s3_valid_q;", "      valid_o <= s3a_valid_q;", "unit-requant_unit"),
 ("requant upper clamp returns the minimum", "rtl/compute/requant_unit.sv", "      q_c = s3_max_q;", "      q_c = s3_min_q - 8'sd1;", "unit-requant_unit"),
 ("act_buffer valid one cycle early", "rtl/mem/act_buffer.sv", "      rd_valid_o <= rd_en_q;", "      rd_valid_o <= rd_en_i;", "unit-act_buffer"),
 ("array drain starts one cycle late", "rtl/compute/systolic_array.sv", "    end else if (drain_trig_c) begin", "    end else if (drain_trig_q) begin", "unit-systolic_array"),
 ("conv prefetch ignores an active beat", "rtl/compute/conv_engine.sv", "(state_q == S_PREFETCH) && !g_active_q && (pf_cnt_q < 3'(BIAS_WORDS));", "(state_q == S_PREFETCH) && (pf_cnt_q < 3'(BIAS_WORDS));", "unit-conv_engine"),
 ("conv tile one beat too long", "rtl/compute/conv_engine.sv", "assign active_c    = kidx_q < k_total_q;", "assign active_c    = kidx_q <= k_total_q;", "unit-conv_engine"),
 ("conv flush window two cycles short", "rtl/compute/conv_engine.sv", "localparam int unsigned FLUSH_CYCLES = 2 * N + ABUF_RD_LAT + RQ_PIPE_LAT;", "localparam int unsigned FLUSH_CYCLES = 2 * N + ABUF_RD_LAT + RQ_PIPE_LAT - 2;", "unit-conv_engine"),
 ("sequencer layer count restarts at 1", "rtl/ctrl/npu_sequencer.sv", "cur_cycles_q <= cur_adv_c ? '0 : cur_cycles_q + 32'd1;", "cur_cycles_q <= cur_adv_c ? 32'd1 : cur_cycles_q + 32'd1;", "unit-npu_sequencer"),
 ("sequencer ends a run without done", "rtl/ctrl/npu_sequencer.sv", "            busy_q  <= 1'b0;\n            done_o  <= 1'b1;", "            busy_q  <= 1'b0;\n            done_o  <= 1'b0;", "unit-npu_sequencer"),
 ("top grants pixels during a DMA", "rtl/npu_top.sv", "assign pix_wr_ready_o = !busy && !dma_busy &&", "assign pix_wr_ready_o = !busy &&", "integ-layers"),
 ("regs STATUS read without this cycle's done", "rtl/host/npu_regs.sv", "err_q | err_i, done_q | done_i, busy_i};", "err_q, done_q, busy_i};", "unit-npu_regs"),
 ("pix_writer writes without the grant", "rtl/chip/pix_writer.sv", "assign wr_en_o   = (state_q == S_FLUSH) && wr_ready_i;", "assign wr_en_o   = (state_q == S_FLUSH);", "unit-pix_writer"),
]

# mutations that need a helper declaration as well: (text, replacement)
EXTRA = {
 "array drain starts one cycle late": ("  logic drain_trig_c;\n", "  logic drain_trig_c;\n  logic drain_trig_q;\n  always_ff @(posedge clk_i) drain_trig_q <= drain_trig_c;\n"),
}


def make(args, timeout=1800):
    """(output, exit code) of make; on a timeout the whole process group (simulator included) is killed."""
    p = subprocess.Popen(["make"] + args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out, p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return "TIMEOUT", None


def run(target, with_assertions):
    return make([target] + ([] if with_assertions else ["VERILATOR_ASSERT="]))[0]


def run_cocotb():
    log, code = make(["cocotb", "COCOTB_ASSERT="])
    if log == "TIMEOUT":
        return "times out"
    errs = sum(int(n) for n in re.findall(r"scoreboard: .*?, (\d+) errors", log))
    if errs:
        return "fails (%d scoreboard errors)" % errs
    if "SimTimeoutError" in log:
        return "fails (test timeout)"
    return "passes" if code == 0 else "fails"


def run_uvm():
    # Verilator turns assertions on by default since 5.04x, so they are switched off explicitly.
    log, code = make(["uvm", "UVM_ASSERT=--no-assert", "UVM_TESTS=npu_full_test"], timeout=2400)
    if log == "TIMEOUT":
        return "times out"
    if "uvm_build.log" in log and "PASS" not in log and "FAIL" not in log:
        return "BUILD ERROR"
    try:
        sim = open(os.path.join(REPO, "build", "logs", "uvm_npu_full_test.log")).read()
    except OSError:
        return "fails (no log)"
    errs = sum(int(n) for n in re.findall(r"^UVM_(?:ERROR|FATAL) :\s*(\d+)", sim, re.M))
    ids = sorted(set(re.findall(r"^UVM_(?:ERROR|FATAL) \S+ @ \d+: \S+ \[([^\]]+)\]", sim, re.M)))
    if errs:
        return "fails (%d UVM errors: %s)" % (errs, ", ".join(ids[:3]))
    return "passes" if code == 0 else "fails"


def main():
    if subprocess.run(["git", "status", "--porcelain", "rtl"], cwd=REPO, capture_output=True, text=True).stdout.strip():
        sys.exit("mutation_run.py: rtl/ has uncommitted changes; refusing to mutate it")
    args = sys.argv[1:]
    with_cocotb = "--cocotb" in args
    uvm_only = "--uvm-only" in args
    with_uvm = uvm_only or "--uvm" in args
    args = [a for a in args if a not in ("--cocotb", "--uvm", "--uvm-only")]
    pick = args[0] if args else ""
    rows = []
    for name, path, old, new, target in M:
        if pick not in name:
            continue
        full = os.path.join(REPO, path)
        orig = open(full).read()
        text = orig
        edits = [(old, new)] + ([EXTRA[name]] if name in EXTRA else [])
        if any(text.count(a) != 1 for a, _ in edits):
            rows.append((name, target, "SETUP ERROR: pattern not found exactly once", "-", "-", "-"))
            continue
        for a, b in edits:
            text = text.replace(a, b)
        try:
            open(full, "w").write(text)
            log_on = "" if uvm_only else run(target, True)
            log_off = "" if uvm_only else run(target, False)
            env = run_cocotb() if with_cocotb else "-"
            uvm = run_uvm() if with_uvm else "-"
        finally:
            open(full, "w").write(orig)
        # only the checkers of tb/sva count; a testbench's own final $fatal is reported the same way
        m = re.search(r"Assertion failed in (\S*\.u_sva\.\S+): (.*)", log_on)
        fired = "%s (%s)" % (m.group(1).split(".")[-1], m.group(2).strip()) if m else "none"
        tb = "passes" if re.search(r"^PASS", log_off, re.M) else ("times out" if log_off == "TIMEOUT" else "fails")
        if uvm_only:
            fired, tb = "-", "-"
        rows.append((name, target, fired, tb, env, uvm))
        print("%-42s | %-18s | %s | testbench alone: %s | cocotb scoreboard alone: %s | UVM env alone: %s" % rows[-1],
              flush=True)
    caught = sum(1 for r in rows if r[2] not in ("none", "-") and not r[2].startswith("SETUP"))
    only = sum(1 for r in rows if r[2] not in ("none", "-") and r[3] == "passes")
    print()
    print("%d mutations, an assertion fails in %d, the testbench alone catches %d, %d seen only by assertions"
          % (len(rows), caught, sum(1 for r in rows if r[3] != "passes" and r[3] != "-"), only))
    if with_cocotb:
        print("the cocotb scoreboard alone catches %d" % sum(1 for r in rows if r[4].startswith("fails")))
    if with_uvm:
        print("the SystemVerilog UVM environment alone catches %d"
              % sum(1 for r in rows if r[5].startswith("fails") or r[5] == "times out"))


if __name__ == "__main__":
    main()
