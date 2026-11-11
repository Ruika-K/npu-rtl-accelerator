#!/usr/bin/env python3
"""Mutation run for the assertion checkers of tb/sva.

Injects one RTL defect at a time, runs the testbench that exercises it twice,
with the assertions and without them (make VERILATOR_ASSERT=), and prints which
assertion fails first and whether the testbench's own checks notice. This is
the evidence that the assertions can fire and what they add; it takes about
50 minutes and is not part of make ci.

Usage: mutation_run.py [substring of a mutation name]
The RTL tree must be clean (git); every file is restored after its run.
"""
import os
import re
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
 ("pix_writer writes without the grant", "rtl/chip/pix_writer.sv", "assign wr_en_o   = (state_q == S_FLUSH) && wr_ready_i;", "assign wr_en_o   = (state_q == S_FLUSH);", "unit-pix_writer"),
]

# mutations that need a helper declaration as well: (text, replacement)
EXTRA = {
 "array drain starts one cycle late": ("  logic drain_trig_c;\n", "  logic drain_trig_c;\n  logic drain_trig_q;\n  always_ff @(posedge clk_i) drain_trig_q <= drain_trig_c;\n"),
}


def run(target, with_assertions):
    cmd = ["make", target] + ([] if with_assertions else ["VERILATOR_ASSERT="])
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def main():
    if subprocess.run(["git", "status", "--porcelain", "rtl"], cwd=REPO, capture_output=True, text=True).stdout.strip():
        sys.exit("mutation_run.py: rtl/ has uncommitted changes; refusing to mutate it")
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    for name, path, old, new, target in M:
        if pick not in name:
            continue
        full = os.path.join(REPO, path)
        orig = open(full).read()
        text = orig
        edits = [(old, new)] + ([EXTRA[name]] if name in EXTRA else [])
        if any(text.count(a) != 1 for a, _ in edits):
            rows.append((name, target, "SETUP ERROR: pattern not found exactly once", "-"))
            continue
        for a, b in edits:
            text = text.replace(a, b)
        try:
            open(full, "w").write(text)
            log_on = run(target, True)
            log_off = run(target, False)
        finally:
            open(full, "w").write(orig)
        m = re.search(r"Assertion failed in (\S+): (.*)", log_on)
        fired = "%s (%s)" % (m.group(1).split(".")[-1], m.group(2).strip()) if m else "none"
        tb = "passes" if re.search(r"^PASS", log_off, re.M) else ("times out" if log_off == "TIMEOUT" else "fails")
        rows.append((name, target, fired, tb))
        print("%-42s | %-18s | %s | testbench alone: %s" % rows[-1], flush=True)
    caught = sum(1 for r in rows if r[2] != "none" and not r[2].startswith("SETUP"))
    only = sum(1 for r in rows if r[2] != "none" and r[3] == "passes")
    print()
    print("%d mutations, an assertion fails in %d, the testbench alone catches %d, %d seen only by assertions"
          % (len(rows), caught, sum(1 for r in rows if r[3] != "passes" and r[3] != "-"), only))


if __name__ == "__main__":
    main()
