# NPU RTL accelerator: single entry point for lint and simulation.
#
#   make lint        Verilator lint, -Wall, every RTL module as its own top
#   make smoke       Top-level reset smoke test with Verilator
#   make ci          lint + smoke (what GitHub Actions runs)
#
#   make compile / make run      VCS flow for TOP and FILELIST
#   make clean
#
# All paths are relative to the repository root. Generated files go to build/.

VERILATOR ?= verilator
VCS       ?= vcs
BUILD_DIR ?= build

LINT_TOPS ?= npu_top
RTL_F     ?= sim/filelists/rtl.f
TOP       ?= npu_top_tb
FILELIST  ?= sim/filelists/tb.f

VERILATOR_LINT_FLAGS ?= --lint-only -Wall --timescale 1ns/1ps
VERILATOR_SIM_FLAGS  ?= --binary --timing -Wall -Wno-fatal --timescale 1ns/1ps -j 0 \
                        --x-assign unique --x-initial unique
VCS_FLAGS ?= -full64 -sverilog -timescale=1ns/1ps -debug_access+all

# lint-% is a pattern rule; GNU make skips pattern search for .PHONY targets,
# so it is deliberately not listed here.
.PHONY: help lint smoke ci compile run clean

help:
	@sed -n '2,9p' Makefile | sed 's/^# \{0,1\}//'

lint: $(addprefix lint-,$(LINT_TOPS))

lint-%:
	$(VERILATOR) $(VERILATOR_LINT_FLAGS) -f $(RTL_F) --top-module $*

smoke:
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f $(FILELIST) --top-module $(TOP) \
	    --Mdir $(BUILD_DIR)/verilator/$(TOP) -o $(TOP)
	$(BUILD_DIR)/verilator/$(TOP)/$(TOP) | tee $(BUILD_DIR)/logs/$(TOP).log
	@grep -q 'PASS' $(BUILD_DIR)/logs/$(TOP).log

ci: lint smoke

compile:
	mkdir -p $(BUILD_DIR)/vcs/$(TOP)
	$(VCS) $(VCS_FLAGS) -f $(FILELIST) -top $(TOP) -o $(BUILD_DIR)/vcs/$(TOP)/simv

run: compile
	$(BUILD_DIR)/vcs/$(TOP)/simv

clean:
	rm -rf $(BUILD_DIR) csrc simv simv.daidir ucli.key vc_hdrs.h DVEfiles
