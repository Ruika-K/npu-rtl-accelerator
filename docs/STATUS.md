# Project Status

Shared handoff between work sessions. Update it after every meaningful change; project facts live here, not in chat history.

## Current milestone

M0 complete: repository skeleton. Next: define the v1 architecture in `docs/` before writing any RTL.

## Completed

- Repository skeleton: RTL / testbench / filelist / docs layout, Verilator-based lint and smoke test in GitHub Actions,
  VCS make targets, engineering conventions in `CONTRIBUTING.md`, MIT license.
- `npu_top` placeholder with a reset smoke test.

## Next steps

1. Write the v1 architecture: goals and non-goals, numeric formats, dataflow, module partition, interfaces.
2. Write the programming model (registers, descriptors) and the verification plan.
3. Pick the first independently verifiable module and implement it with a self-checking testbench.

## Known limitations

- The open flow is Verilator; a commercial simulator (VCS) is available to the author on a separate machine and is used
  as an additional check only. Results are recorded here; no files are transferred from that machine.

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-07-13 | Use the `.sv` extension for SystemVerilog | Unambiguous language, simulator friendly |
| 2026-07-13 | Keep RTL and testbenches in separate trees | Clear synthesis boundary, separate organization of verification code |
| 2026-07-13 | The local repository is the single source of truth | Nothing is taken from the machine that hosts the commercial simulator |

## VCS run log

| Date | Commit | VCS version | Test | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| - | - | - | - | Not yet run | - |
