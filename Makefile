# NPU RTL accelerator: single entry point for lint and simulation.
#
#   make lint        Verilator lint, -Wall, every RTL module as its own top
#   make model-test  Python golden model self-test
#   make vectors     Generate Python golden vectors used by the unit testbenches
#   make unit        Build and run every unit testbench with Verilator
#   make unit-pe     One unit testbench (unit-<module>)
#   make smoke       Top-level reset smoke test with Verilator
#   make ci          lint + model-test + unit + smoke (what GitHub Actions runs)
#
#   make compile / make run      VCS flow for TOP and FILELIST (commercial simulator)
#   make vcs-unit-<module>       VCS build and run of one unit testbench
#   make clean
#
# All paths are relative to the repository root. Generated files go to build/.

VERILATOR ?= verilator
VCS       ?= vcs
PYTHON    ?= python3
BUILD_DIR ?= build
SEED      ?= 1

# Units: RTL module names that have tb/unit/<module>_tb.sv and
# sim/filelists/unit_<module>.f
UNITS     ?= requant_unit pe systolic_array axil_slave npu_regs npu_sequencer act_buffer conv_engine \
             pool_engine argmax_engine
LINT_TOPS ?= npu_top requant_unit pe systolic_array axil_slave npu_regs npu_sequencer \
             sram_1r1w act_buffer weight_buffer host_mem_adapter conv_engine pool_engine argmax_engine

RTL_F     ?= sim/filelists/rtl.f
TOP       ?= npu_top_tb
FILELIST  ?= sim/filelists/tb.f

REQUANT_N ?= 100000
GEMM_N    ?= 30
DESC_N    ?= 2000
LAYER_N   ?= 24
VEC_DIR   := $(BUILD_DIR)/vectors
REQUANT_VECTORS := $(VEC_DIR)/requant_s$(SEED)_n$(REQUANT_N).txt
GEMM_VECTORS    := $(VEC_DIR)/gemm_s$(SEED)_n$(GEMM_N).txt
DESC_VECTORS    := $(VEC_DIR)/desc_s$(SEED)_n$(DESC_N).txt
LAYER_VECTORS   := $(VEC_DIR)/layers_s$(SEED)_n$(LAYER_N).txt
PLUSARGS  ?= +seed=$(SEED) +vectors=$(abspath $(REQUANT_VECTORS)) +gemm_vectors=$(abspath $(GEMM_VECTORS)) \
             +desc_vectors=$(abspath $(DESC_VECTORS)) +layer_vectors=$(abspath $(LAYER_VECTORS))

VERILATOR_LINT_FLAGS ?= --lint-only -Wall --timescale 1ns/1ps
VERILATOR_SIM_FLAGS  ?= --binary --timing -Wall -Wno-fatal --timescale 1ns/1ps -j 0 \
                        --x-assign unique --x-initial unique
VCS_FLAGS ?= -full64 -sverilog -timescale=1ns/1ps -debug_access+all

# lint-%, unit-% and vcs-unit-% are pattern rules; GNU make skips pattern
# search for .PHONY targets, so they are deliberately not listed here.
.PHONY: help lint model-test vectors unit smoke ci compile run clean

help:
	@sed -n '2,16p' Makefile | sed 's/^# \{0,1\}//'

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
lint: $(addprefix lint-,$(LINT_TOPS))

lint-%:
	$(VERILATOR) $(VERILATOR_LINT_FLAGS) -f $(RTL_F) --top-module $*

# ---------------------------------------------------------------------------
# Python golden model
# ---------------------------------------------------------------------------
model-test:
	$(PYTHON) model/test_npu_ref.py

vectors: $(REQUANT_VECTORS) $(GEMM_VECTORS) $(DESC_VECTORS) $(LAYER_VECTORS)

$(LAYER_VECTORS): model/gen_layer_vectors.py model/npu_ref.py
	mkdir -p $(VEC_DIR)
	$(PYTHON) model/gen_layer_vectors.py --n $(LAYER_N) --seed $(SEED) -o $@

$(DESC_VECTORS): model/gen_desc_vectors.py model/npu_ref.py
	mkdir -p $(VEC_DIR)
	$(PYTHON) model/gen_desc_vectors.py --n $(DESC_N) --seed $(SEED) -o $@

$(REQUANT_VECTORS): model/gen_requant_vectors.py model/npu_ref.py
	mkdir -p $(VEC_DIR)
	$(PYTHON) model/gen_requant_vectors.py --n $(REQUANT_N) --seed $(SEED) -o $@

$(GEMM_VECTORS): model/gen_gemm_vectors.py model/npu_ref.py
	mkdir -p $(VEC_DIR)
	$(PYTHON) model/gen_gemm_vectors.py --n $(GEMM_N) --seed $(SEED) -o $@

# ---------------------------------------------------------------------------
# Verilator simulation
# ---------------------------------------------------------------------------
unit: $(addprefix unit-,$(UNITS))

unit-%: vectors
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/unit_$*.f --top-module $*_tb \
	    --Mdir $(BUILD_DIR)/verilator/$*_tb -o $*_tb
	$(BUILD_DIR)/verilator/$*_tb/$*_tb $(PLUSARGS) | tee $(BUILD_DIR)/logs/$*_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/$*_tb.log

smoke:
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f $(FILELIST) --top-module $(TOP) \
	    --Mdir $(BUILD_DIR)/verilator/$(TOP) -o $(TOP)
	$(BUILD_DIR)/verilator/$(TOP)/$(TOP) | tee $(BUILD_DIR)/logs/$(TOP).log
	@grep -q 'PASS' $(BUILD_DIR)/logs/$(TOP).log

ci: lint model-test unit smoke

# ---------------------------------------------------------------------------
# VCS (commercial simulator, optional)
# ---------------------------------------------------------------------------
compile:
	mkdir -p $(BUILD_DIR)/vcs/$(TOP)
	$(VCS) $(VCS_FLAGS) -f $(FILELIST) -top $(TOP) -o $(BUILD_DIR)/vcs/$(TOP)/simv

run: compile
	$(BUILD_DIR)/vcs/$(TOP)/simv

vcs-unit-%: vectors
	$(MAKE) compile TOP=$*_tb FILELIST=sim/filelists/unit_$*.f
	$(BUILD_DIR)/vcs/$*_tb/simv $(PLUSARGS)

clean:
	rm -rf $(BUILD_DIR) csrc simv simv.daidir ucli.key vc_hdrs.h DVEfiles
