# NPU RTL accelerator: single entry point for lint and simulation.
#
#   make lint        Verilator lint, -Wall, every RTL module as its own top, and the SVA checkers
#   make model-test  Python golden model self-test
#   make vectors     Generate Python golden vectors used by the unit testbenches
#   make unit        Build and run every unit testbench with Verilator
#   make unit-pe     One unit testbench (unit-<module>)
#   make integ       Layer cases, the LeNet-5 networks (AXI4-Lite) and chip_top (APB + pixel stream)
#   make integ-mnist One network (integ-<net>, IMAGES=<n> to limit the images)
#   make cocotb      pyuvm environment for npu_top (tb/cocotb, needs tb/cocotb/requirements.txt)
#   make uvm         SystemVerilog UVM environment (tb/uvm), every test (Verilator 5.052+, VERILATOR_UVM)
#   make uvm-elab    SystemVerilog UVM environment compiled and elaborated with slang
#   make cover       Cover properties of tb/sva merged over all runs; fails if one was never reached
#   make ci          lint + model-test + unit + integ + cocotb + uvm-elab + cover (what GitHub Actions runs)
#   make synth       sv2v + Yosys (xc7 resources) and nextpnr-ecp5 (Fmax), see synth/README.md
#
#   make compile / make run      VCS flow for TOP and FILELIST (commercial simulator)
#   make vcs-unit-<module>       VCS build and run of one unit testbench
#   make vcs-uvm                 VCS build of tb/uvm and every UVM test (UVM_TESTS, VCS_UVM)
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
# Testbenches are built with one main() (tb/common/verilator_main.cpp, model
# class Vtop) that also writes the cover-property database; --assert enables
# the assertions bound from tb/sva (VERILATOR_ASSERT= builds without them, which
# tb/sva/mutation_run.py uses as its control).
VERILATOR_ASSERT     ?= --assert
VERILATOR_SIM_FLAGS  ?= --cc --exe --build --timing --prefix Vtop -Wall -Wno-fatal --timescale 1ns/1ps -j 0 \
                        --x-assign unique --x-initial unique $(VERILATOR_ASSERT) --coverage-user \
                        $(abspath tb/common/verilator_main.cpp)
COV_DIR ?= $(BUILD_DIR)/cov
VCS_FLAGS ?= -full64 -sverilog -timescale=1ns/1ps -debug_access+all

# lint-%, unit-% and vcs-unit-% are pattern rules; GNU make skips pattern
# search for .PHONY targets, so they are deliberately not listed here.
.PHONY: help lint lint-synth-wrapper lint-sva model-test vectors unit integ integ-layers integ-chip cocotb uvm uvm-elab vcs-uvm cover ci synth synth-xilinx synth-ecp5 compile run clean

help:
	@sed -n '2,18p' Makefile | sed 's/^# \{0,1\}//'

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
lint: $(addprefix lint-,$(LINT_TOPS)) lint-synth-wrapper lint-sva

# The assertion checkers of tb/sva, bound into the whole design. disable iff
# reads the asynchronous resets synchronously, hence -Wno-SYNCASYNCNET.
lint-sva:
	$(VERILATOR) $(VERILATOR_LINT_FLAGS) --assert -Wno-SYNCASYNCNET -f $(LINT_F) -f sim/filelists/sva.f -f sim/filelists/sva_chip.f \
	    --top-module chip_top

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
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs $(COV_DIR)
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/unit_$*.f --top-module $*_tb \
	    --Mdir $(BUILD_DIR)/verilator/$*_tb -o $*_tb
	$(BUILD_DIR)/verilator/$*_tb/$*_tb $(PLUSARGS) +cov=$(COV_DIR)/unit_$*.dat | tee $(BUILD_DIR)/logs/$*_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/$*_tb.log

# Network tests: model/data/<net>/program.txt is committed, generated by model/quantize.py
NETS      ?= mnist cifar10   # networks with committed model/data/<net>/program.txt
IMAGES    ?= 100

integ: integ-layers $(addprefix integ-,$(NETS)) integ-chip

# chip_top: APB bridge, pixel stream across clock domains, clock gate, with the first MNIST images
integ-chip:
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs $(COV_DIR)
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_chip_top.f --top-module chip_top_tb \
	    --Mdir $(BUILD_DIR)/verilator/chip_top_tb -o chip_top_tb
	$(BUILD_DIR)/verilator/chip_top_tb/chip_top_tb +program=$(abspath model/data/mnist/program.txt) +images=4 \
	    +cov=$(COV_DIR)/integ_chip.dat | tee $(BUILD_DIR)/logs/chip_top_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/chip_top_tb.log

integ-layers: vectors
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs $(COV_DIR)
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_npu_top.f --top-module npu_top_tb \
	    --Mdir $(BUILD_DIR)/verilator/npu_top_tb -o npu_top_tb
	$(BUILD_DIR)/verilator/npu_top_tb/npu_top_tb $(PLUSARGS) +cov=$(COV_DIR)/integ_layers.dat | tee $(BUILD_DIR)/logs/npu_top_tb.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/npu_top_tb.log

$(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb: $(shell cat sim/filelists/rtl.f sim/filelists/sva.f sim/filelists/tb_common.f) tb/integration/npu_lenet_tb.sv tb/common/verilator_main.cpp
	mkdir -p $(BUILD_DIR)/verilator $(BUILD_DIR)/logs
	$(VERILATOR) $(VERILATOR_SIM_FLAGS) -f sim/filelists/integ_npu_lenet.f --top-module npu_lenet_tb \
	    --Mdir $(BUILD_DIR)/verilator/npu_lenet_tb -o npu_lenet_tb

integ-%: $(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb
	mkdir -p $(BUILD_DIR)/logs $(COV_DIR)
	$(BUILD_DIR)/verilator/npu_lenet_tb/npu_lenet_tb +program=$(abspath model/data/$*/program.txt) +images=$(IMAGES) \
	    +cov=$(COV_DIR)/integ_$*.dat | tee $(BUILD_DIR)/logs/npu_lenet_$*.log
	@grep -q '^PASS' $(BUILD_DIR)/logs/npu_lenet_$*.log

# VCS variants of the network tests
vcs-integ-%: 
	$(MAKE) compile TOP=npu_lenet_tb FILELIST=sim/filelists/integ_npu_lenet.f
	$(BUILD_DIR)/vcs/npu_lenet_tb/simv +program=$(abspath model/data/$*/program.txt) +images=$(IMAGES)

# cocotb + pyuvm environment for npu_top: random layers and programs, errors and
# DMA, checked by a register-level reference model, with functional-coverage
# closure as its last test. The SVA checkers run in it too; its cover database
# joins the others for make cover.
COCOTB_ASSERT ?= --assert
cocotb:
	mkdir -p $(COV_DIR) $(BUILD_DIR)/cocotb $(BUILD_DIR)/logs
	rm -f $(BUILD_DIR)/cocotb/results.xml tb/cocotb/coverage.dat
	$(MAKE) -C tb/cocotb COCOTB_ASSERT="$(COCOTB_ASSERT)" 2>&1 | tee $(BUILD_DIR)/logs/cocotb.log
	@if [ -f tb/cocotb/coverage.dat ]; then mv tb/cocotb/coverage.dat $(COV_DIR)/cocotb.dat; fi
	python3 tb/cocotb/check_results.py $(BUILD_DIR)/cocotb/results.xml

# SystemVerilog UVM environment for npu_top (tb/uvm), on the Accellera library.
#   uvm       builds it with Verilator 5.052 or later (VERILATOR_UVM; the first
#             releases that run UVM) and runs every test in UVM_TESTS
#   uvm-elab  compiles and elaborates it with slang (any warning fails)
#   vcs-uvm   the same tests on VCS
UVM_CORE_TAG ?= 2020.3.1
UVM_SRC      ?= $(BUILD_DIR)/uvm-core/src
UVM_TESTS    ?= npu_reg_reset_test npu_reg_bit_bash_test npu_mem_walk_test npu_layer_test npu_desc_err_test \
                npu_dma_test npu_lenet_test npu_full_test
UVM_PLUSARGS ?= +layer_vectors=$(abspath $(LAYER_VECTORS)) +desc_vectors=$(abspath $(DESC_VECTORS)) \
                +program=$(abspath model/data/mnist/program.txt) +images=$(IMAGES) +UVM_NO_RELNOTES
# uvm-1.2, or uvm-ieee-2020 on VCS releases that ship IEEE 1800.2-2020
VCS_UVM      ?= uvm-1.2

VERILATOR_UVM ?= $(VERILATOR)
# --no-assert compiles the bound checkers out (Verilator 5.052+ enables assertions by default)
UVM_ASSERT    ?= --assert
UVM_VL_DIR    := $(BUILD_DIR)/verilator/npu_uvm
UVM_ABS       := $(abspath $(UVM_SRC))

# Runs each test of UVM_TESTS with the simulator $(1); a test passes on its TEST PASSED line.
define run_uvm_tests
	@for t in $(UVM_TESTS); do \
	  $(1) +UVM_TESTNAME=$$t $(UVM_PLUSARGS) > $(BUILD_DIR)/logs/uvm_$$t.log 2>&1; \
	  if grep -q 'TEST PASSED' $(BUILD_DIR)/logs/uvm_$$t.log; then \
	    echo "PASS $$t: $$(grep -h -o -E '(checked|[0-9]+ (images|layer cases|error vectors)).*' $(BUILD_DIR)/logs/uvm_$$t.log | tr '\n' ' ')"; \
	  else echo "FAIL $$t (see $(BUILD_DIR)/logs/uvm_$$t.log)"; grep -E '^UVM_(ERROR|FATAL) ' $(BUILD_DIR)/logs/uvm_$$t.log | head -5; exit 1; fi; \
	done
endef

$(UVM_SRC)/uvm_pkg.sv:
	rm -rf $(BUILD_DIR)/uvm-core
	git clone --quiet --depth 1 --branch $(UVM_CORE_TAG) https://github.com/accellera-official/uvm-core.git $(BUILD_DIR)/uvm-core

uvm-elab: $(UVM_SRC)/uvm_pkg.sv
	python3 tb/uvm/slang_check.py --uvm $(UVM_SRC)

vcs-uvm: vectors
	mkdir -p $(BUILD_DIR)/vcs/uvm $(BUILD_DIR)/logs
	$(VCS) $(VCS_FLAGS) -ntb_opts $(VCS_UVM) -f sim/filelists/uvm.f -top npu_uvm_top -o $(BUILD_DIR)/vcs/uvm/simv
	$(call run_uvm_tests,$(BUILD_DIR)/vcs/uvm/simv +ntb_random_seed=$(SEED))

# The HDL backdoor of the UVM DPI library has no Verilator backend and is not
# used: +define+UVM_HDL_NO_DPI, and tb/uvm/uvm_dpi_verilator.cc compiles the rest.
uvm: vectors $(UVM_SRC)/uvm_pkg.sv
	mkdir -p $(UVM_VL_DIR) $(BUILD_DIR)/logs
	$(VERILATOR_UVM) --version
	$(VERILATOR_UVM) --binary --vpi --timing -j 0 -Wno-fatal -Wno-lint -Wno-style $(UVM_ASSERT) --timescale 1ns/1ps \
	    +define+UVM_HDL_NO_DPI +incdir+$(UVM_ABS) $(UVM_ABS)/uvm_pkg.sv -f sim/filelists/uvm.f --top-module npu_uvm_top \
	    --Mdir $(UVM_VL_DIR) -o npu_uvm $(abspath tb/uvm/uvm_dpi_verilator.cc) -CFLAGS -I$(UVM_ABS)/dpi \
	    > $(BUILD_DIR)/logs/uvm_build.log 2>&1 || { tail -30 $(BUILD_DIR)/logs/uvm_build.log; exit 1; }
	$(call run_uvm_tests,$(UVM_VL_DIR)/npu_uvm +verilator+seed+$(SEED))

# Cover-property closure: every cover property of tb/sva must have been reached
# by at least one of the runs above (their databases are in $(COV_DIR)), or be
# waived with a reason in tb/sva/cover_waivers.txt.
cover:
	python3 tb/sva/cover_report.py $(COV_DIR)/*.dat --waivers tb/sva/cover_waivers.txt

ci: lint model-test unit integ cocotb uvm-elab cover

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
	    || { python3 synth/nextpnr_summary.py $(SYNTH_DIR)/nextpnr.log $(SYNTH_DIR)/npu_top_ooc.v; exit 1; }
	@python3 synth/nextpnr_summary.py $(SYNTH_DIR)/nextpnr.log $(SYNTH_DIR)/npu_top_ooc.v

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
