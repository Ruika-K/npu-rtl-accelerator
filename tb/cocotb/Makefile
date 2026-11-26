# cocotb + pyuvm environment for npu_top (docs/verification.md section 2.4).
# Run from the repository root with `make cocotb`, or here with `make`.
SIM            ?= verilator
TOPLEVEL_LANG  := verilog
TOPLEVEL       := npu_top
MODULE         ?= test_npu

REPO           := $(abspath ../..)
VERILOG_SOURCES := $(addprefix $(REPO)/,$(shell grep -v '^\#' $(REPO)/sim/filelists/rtl.f) \
                                        $(shell cat $(REPO)/sim/filelists/sva.f))
SIM_BUILD      := $(REPO)/build/cocotb/sim_build
COCOTB_RESULTS_FILE := $(REPO)/build/cocotb/results.xml
COCOTB_ASSERT  ?= --assert
EXTRA_ARGS     += $(COCOTB_ASSERT) --coverage-user -Wno-fatal --timescale 1ns/1ps -Wno-lint -Wno-style
export PYTHONPATH := $(REPO)/model:$(abspath .):$(PYTHONPATH)
export NPU_COV_YAML ?= $(REPO)/build/cocotb/functional_coverage.yml

include $(shell cocotb-config --makefiles)/Makefile.sim
