"""pyuvm tests for npu_top (make cocotb).

Every test builds the environment, resets the DUT and runs one sequence; the
scoreboard checks every bus transaction against the reference model. The last
test checks functional-coverage closure over all tests of the run.
"""
import os
import random

import cocotb
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotb_coverage.coverage import coverage_db
from pyuvm import ConfigDB, uvm_root, uvm_test

from npu_env import coverage
from npu_env.axi_mem import AxiMem
from npu_env.env import NpuEnv
from npu_env.sequences import DmaSeq, ErrorSeq, ProgramSeq, RandomLayerSeq

SEED = int(os.environ.get("NPU_SEED", "1"))
# Simulated time per test (the longest needs under 1 ms): a DUT that stops
# answering fails its test instead of hanging the run.
TEST_TIMEOUT_MS = 3


class NpuTest(uvm_test):
    seq_class = None
    seq_args = {}

    def build_phase(self):
        self.dut = cocotb.top
        self.ext = AxiMem(self.dut, seed=SEED)
        ConfigDB().set(None, "*", "DUT", self.dut)
        ConfigDB().set(None, "*", "EXT_MEM", self.ext)
        self.env = NpuEnv("env", self)

    async def run_phase(self):
        self.raise_objection()
        d = self.dut
        cocotb.start_soon(Clock(d.clk_i, 10, units="ns").start())
        d.rst_ni.value = 0
        d.pix_wr_en_i.value = 0
        d.pix_wr_addr_i.value = 0
        d.pix_wr_data_i.value = 0
        d.pix_wr_be_i.value = 0
        self.ext.start(cocotb.start_soon)
        await ClockCycles(d.clk_i, 5)
        d.rst_ni.value = 1
        await ClockCycles(d.clk_i, 2)
        seq = self.seq_class(rng=random.Random(f"{SEED}-{type(self).__name__}"), **self.seq_args)
        await seq.start(self.env.agent.seqr)
        await ClockCycles(d.clk_i, 10)
        self.drop_objection()


@pyuvm.test(timeout_time=TEST_TIMEOUT_MS, timeout_unit="ms")
class RandomLayerTest(NpuTest):
    seq_class = RandomLayerSeq
    seq_args = {"n": 40}


@pyuvm.test(timeout_time=TEST_TIMEOUT_MS, timeout_unit="ms")
class ProgramTest(NpuTest):
    seq_class = ProgramSeq


@pyuvm.test(timeout_time=TEST_TIMEOUT_MS, timeout_unit="ms")
class ErrorTest(NpuTest):
    seq_class = ErrorSeq


@pyuvm.test(timeout_time=TEST_TIMEOUT_MS, timeout_unit="ms")
class DmaTest(NpuTest):
    seq_class = DmaSeq


@cocotb.test()
async def coverage_closure(dut):
    """Every coverage bin of npu_env/coverage.py was hit by the tests above."""
    out = os.environ.get("NPU_COV_YAML")
    if out:
        coverage_db.export_to_yaml(filename=out)
    total, hit, missed = coverage.closure()
    dut._log.info(f"functional coverage: {hit} of {total} bins")
    for name, holes in missed:
        dut._log.error(f"missed: {name} {holes}")
    assert not missed, f"{len(missed)} coverage items not closed"
