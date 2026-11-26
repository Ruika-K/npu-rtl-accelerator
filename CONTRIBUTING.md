# Contributing and Engineering Conventions

This file is the working agreement for every change in this repository.

## Before any change

1. Read `README.md`.
2. Read `docs/STATUS.md` for the current milestone, open issues and next step.
3. Read the relevant section of `docs/architecture.md` or `docs/verification.md` before editing RTL.
4. Inspect the working tree and preserve unrelated changes.

## Repository layout

- `rtl/`: synthesizable SystemVerilog only.
- `tb/`: testbenches, assertions, reference functions.
- `model/`: Python golden model, vector generators, quantization and training scripts.
- `sim/filelists/`: simulator filelists with paths relative to the repository root, shared by Verilator and VCS.
- `docs/`: architecture, programming model, verification plan, decisions, status.
- `build/`: generated output; never committed.

## RTL conventions

- Files are `.sv` for SystemVerilog and `.svh` for headers. One primary module per file; the file name equals the module name.
- Use the synthesizable SystemVerilog subset: `logic`, `always_ff`, `always_comb`, packages, packed structs and enums.
- Active-low resets end in `_ni`; inputs end in `_i`; outputs end in `_o`.
- Put `` `default_nettype none`` at the start of every RTL and testbench file and restore `wire` at the end.
- Document an interface in `docs/` before implementing it. Derive interfaces from the project's own documented requirements.
- Every functional RTL change comes with a self-checking testbench or a clearly documented verification follow-up.
- RTL must lint clean with `verilator --lint-only -Wall`; a `lint_off` pragma needs a comment with the reason.

## Testbench conventions

- Stay within the SystemVerilog subset Verilator 5 supports: procedural testbenches, queues, `$fopen` / `$fscanf`; no classes, no UVM.
- Drive stimulus on the falling clock edge with blocking assignments; check on the falling edge. The DUT then samples
  identical values under Verilator and VCS.
- Every testbench prints exactly one `PASS:` or `FAIL:` line and exits with `$finish` or `$fatal`.
- Reference functions shared between testbenches live in `tb/common/npu_tb_pkg.sv` and mirror `model/npu_ref.py`.
- Assertions live in `tb/sva/<module>_sva.sv`, one checker per RTL module, attached with `bind`; the RTL files
  contain no assertions. Use `m_` for environment assumptions, `a_` for guarantees and `c_` for cover properties,
  and stay inside the SVA subset of the CI simulator (`docs/verification.md` section 2.3). A new cover property
  must be reached by a test, or waived with a reason in `tb/sva/cover_waivers.txt`.
- Each unit testbench has a filelist `sim/filelists/unit_<module>.f` and is run with `make unit-<module>`.

## Golden model

- `model/npu_ref.py` uses Python integers only and is the single source of expected values.
- Hand-computed cases live in `model/test_npu_ref.py` and are repeated in the matching testbench.

## Git workflow

- One branch per milestone (`feat/m2-systolic-array`, `docs/...`), merged into `main` through a pull request with squash merge.
- `make ci` must pass before a pull request is opened.
- Commit messages follow `type(scope): summary` with an explanatory body; English throughout.
- Update `docs/STATUS.md` in every pull request: completed work, verification evidence, next step.
- Never commit `build/`, simulator output, credentials, or third-party proprietary material.
