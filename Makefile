VCS       ?= vcs
TOP       ?= npu_top_tb
BUILD_DIR ?= build/vcs
FILELIST  ?= sim/filelists/tb.f

VCS_FLAGS ?= -full64 -sverilog -timescale=1ns/1ps -debug_access+all

.PHONY: help compile run clean

help:
	@echo "make compile  Compile the testbench with VCS"
	@echo "make run      Run the compiled smoke test"
	@echo "make clean    Remove generated files"

compile:
	mkdir -p $(BUILD_DIR)
	$(VCS) $(VCS_FLAGS) -f $(FILELIST) -top $(TOP) -o $(BUILD_DIR)/simv

run: compile
	$(BUILD_DIR)/simv

clean:
	rm -rf build csrc simv simv.daidir ucli.key vc_hdrs.h DVEfiles
