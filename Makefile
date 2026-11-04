# NPU RTL accelerator: single entry point for lint and simulation.
#
#   make lint        Verilator lint, -Wall, every RTL module as its own top
#   make model-test  Python golden model self-test
#   make vectors     Generate Python golden vectors used by the unit testbenches
#   make unit        Build and run every unit testbench with Verilator
#   make unit-pe     One unit testbench (unit-<module>)
#   make integ       Layer cases, the LeNet-5 networks (AXI4-Lite) and chip_top (APB + pixel stream)
#   make integ-mnist One network (integ-<net>, IMAGES=<n> to limit the images)
#   make ci          lint + model-test + unit + integ (what GitHub Actions runs)
#   make synth       sv2v + Yosys (xc7 resources) and nextpnr-ecp5 (Fmax), see synth/README.md
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
             pool_engine argmax_engine async_fifo pix_writer
LINT_TOPS ?= npu_top requant_unit pe systolic_array axil_slave npu_regs npu_sequencer \
             sram_1r1w act_buffer weight_buffer host_mem_adapter conv_engine pool_engine argmax_engine \
             npu_dma cdc_sync async_fifo rst_sync clk_gate apb2axil pix_writer chip_regs chip_top

RTL_F     ?= sim/filelists/rtl.f        # npu_top and below (synthesis, integration tests)
LINT_F    ?= sim/filelists/rtl_chip.f   # everything, including the chip-level blocks
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
.PHONY: help lint lint-synth-wrapper model-test vectors unit integ integ-layers integ-chip ci synth synth-xilinx synth-ecp5 compile run clean

help:
	@sed -n '2,16p' Makefile | sed 's/^# \{0,1\}//'

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
lint: $(addprefix lint-,$(LINT_TOPS)) lint-synth-wrapper

lint-synth-wrapper:
	$(VERILATOR) $(VERILATOR_LINT_FLAGS) -f $(RTL_F) synth/npu_top_ooc.sv --top-module npu_top_ooc

lint-%:
	$(VERILATOR) $(VERILATOR_LINT_FLAGS) -f $(LINT_F) --top-module $*

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

# Network tests: model/data/<net>/program.txt is committed, generated by model/quantize.py
NETS      ?= mnist    # networks with committed model/data/<net>/program.txt
IMAGES    ?= 100

integ: integ-layers $(addprefix integ-,$(NETS)) integ-chip

# chip_top: APB bridge, pixel stream across clock domains, clock gate, with the first MNIST images
integ-chip:
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_chip_top.f --top-module chip_top_tb \
	    --Mdir $(BUILD_DIR)/verilator/chip_top_tb -o chip_top_tb
	$(BUILD_DIR)/verilator/chip_top_tb/chip_top_tb +program=$(abspath model/data/mnist/program.txt) +images=4 \
	    | tee $(BUILD_DIR)/logs/chip_top_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/chip_top_tb.log

integ-layers: vectors
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_npu_top.f --top-module npu_top_tb \
	    --Mdir $(BUILD_DIR)/verilator/npu_top_tb -o npu_top_tb
	$(BUILD_DIR)/verilator/npu_top_tb/npu_top_tb $(PLUSARGS) | tee $(BUILD_DIR)/logs/npu_top_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/npu_top_tb.log

$(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb: $(shell cat sim/filelists/rtl.f sim/filelists/tb_common.f) tb/integration/npu_lenet_tb.sv
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_npu_lenet.f --top-module npu_lenet_tb \
	    --Mdir $(BUILD_DIR)/verilator/npu_lenet_tb -o npu_lenet_tb

integ-%: $(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb
	mkdir -p $(BUILD_DIR)/logs
	$(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb +program=$(abspath model/data/$*/program.txt) +images=$(IMAGES) \
	    | tee $(BUILD_DIR)/logs/npu_lenet_$*.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/npu_lenet_$*.log

# VCS variants of the network tests
vcs-integ-%: 
	$(MAKE) compile TOP=npu_lenet_tb FILELIST=sim/filelists/integ_npu_lenet.f
	$(BUILD_DIR)/vcs/npu_lenet_tb/simv +program=$(abspath model/data/$*/program.txt) +images=$(IMAGES)

ci: lint model-test unit integ

# ---------------------------------------------------------------------------
# Synthesis (open source: sv2v + yosys, nextpnr-ecp5 for timing), see synth/README.md
# ---------------------------------------------------------------------------
SV2V         ?= sv2v
YOSYS        ?= yosys
NEXTPNR_ECP5 ?= nextpnr-ecp5
PNR_SEED     ?= 1          # nextpnr placement seed; CI sweeps several
SYNTH_DIR    := $(BUILD_DIR)/synth
RTL_SRCS     := $(shell grep -v '^\#' $(RTL_F))

$(SYNTH_DIR)/npu_top.v: $(RTL_SRCS)
	mkdir -p $(SYNTH_DIR)
	$(SV2V) --top npu_top $(RTL_SRCS) > $@

synth-xilinx: $(SYNTH_DIR)/npu_top.v
	$(YOSYS) -q -l $(SYNTH_DIR)/xilinx.log -s synth/xilinx.ys
	@python3 synth/stat_table.py $(SYNTH_DIR)/xilinx_stat.txt

# ECP5 place and route runs on the out-of-context wrapper (see synth/npu_top_ooc.sv)
$(SYNTH_DIR)/npu_top_ooc.v: $(RTL_SRCS) synth/npu_top_ooc.sv
	mkdir -p $(SYNTH_DIR)
	$(SV2V) --top npu_top_ooc $(RTL_SRCS) synth/npu_top_ooc.sv > $@

synth-ecp5: $(SYNTH_DIR)/npu_top_ooc.v
	$(YOSYS) -q -l $(SYNTH_DIR)/ecp5.log -s synth/ecp5.ys
	$(NEXTPNR_ECP5) --85k --package CABGA381 --speed 8 --json $(SYNTH_DIR)/npu_top_ooc_ecp5.json \
	    --freq 100 --timing-allow-fail --seed $(PNR_SEED) --router router2 --log $(SYNTH_DIR)/nextpnr.log > /dev/null 2>&1 \
	    || { python3 synth/nextpnr_summary.py $(SYNTH_DIR)/nextpnr.log; exit 1; }
	@python3 synth/nextpnr_summary.py $(SYNTH_DIR)/nextpnr.log

synth: synth-xilinx synth-ecp5

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
