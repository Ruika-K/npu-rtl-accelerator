#!/usr/bin/env python3
"""Compile and elaborate the UVM testbench with slang (pyslang).

No open-source simulator in CI runs this environment, so this is the CI gate
for it: the whole design (RTL, bound SVA checkers, UVM environment, tests) is
parsed, type-checked and elaborated against the Accellera UVM library, with
every warning treated as an error. Running the tests needs a UVM simulator
(see the vcs-uvm target).

    tb/uvm/slang_check.py --uvm <uvm-core>/src [-f sim/filelists/uvm.f]
"""
import argparse
import sys

from pyslang.driver import CommandLineOptions, Driver


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uvm", required=True, help="UVM library source directory (contains uvm_pkg.sv)")
    ap.add_argument("-f", dest="filelist", default="sim/filelists/uvm.f")
    ap.add_argument("--top", default="npu_uvm_top")
    args = ap.parse_args()

    # One compilation unit, so macros carry across files as they do in VCS.
    # The only warning the UVM library itself raises is an unknown '\.'
    # escape in a string; every other warning fails the check.
    cmd = ["slang", f"+incdir+{args.uvm}", f"{args.uvm}/uvm_pkg.sv", "-f", args.filelist, "--top", args.top,
           "--timescale", "1ns/1ps", "-Wno-unknown-escape-code", "--error-limit", "50", "--single-unit"]
    d = Driver()
    d.addStandardArgs()
    if not d.parseCommandLine(" ".join(cmd), CommandLineOptions()) or not d.processOptions():
        sys.exit("slang: bad command line")
    ok = d.parseAllSources()
    ok = d.runFullCompilation(False) and ok
    errors, warnings = d.diagEngine.numErrors, d.diagEngine.numWarnings
    print(f"slang: {errors} errors, {warnings} warnings ({args.top}, UVM from {args.uvm})")
    sys.exit(0 if ok and errors == 0 and warnings == 0 else 1)


if __name__ == "__main__":
    main()
