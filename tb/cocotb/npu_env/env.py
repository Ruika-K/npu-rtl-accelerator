"""The npu_top environment: AXI4-Lite agent, scoreboard with the reference model."""
from pyuvm import uvm_env

from .axil import AxilAgent
from .scoreboard import NpuScoreboard


class NpuEnv(uvm_env):
    def build_phase(self):
        self.agent = AxilAgent("axil", self)
        self.scoreboard = NpuScoreboard("scoreboard", self)

    def connect_phase(self):
        self.agent.mon.ap.connect(self.scoreboard.analysis_export)
