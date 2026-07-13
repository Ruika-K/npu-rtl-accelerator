# Project Instructions

## Context

- This repository is a clean-room, from-scratch SystemVerilog NPU learning project.
- The local repository is the sole source of truth.
- VCS is available only on a restricted training server. Code may be copied into that server, but nothing should be extracted from it.
- Never add training-provider source code, confidential specifications, credentials, generated simulator output, or text copied from restricted systems.

## Start Every Task Here

1. Read `README.md`.
2. Read `docs/STATUS.md`.
3. Read the relevant architecture or verification document before editing RTL.
4. Inspect the working tree and preserve unrelated user changes.

## Repository Layout

- `rtl/`: synthesizable SystemVerilog only.
- `tb/`: testbenches, assertions, scoreboards, and reference-model integration.
- `sim/filelists/`: simulator file lists, using paths relative to the repository root.
- `docs/`: architecture, verification plan, decisions, and shared project status.
- `build/`: generated output; never commit it.

## Engineering Conventions

- Use `.sv` for SystemVerilog and `.svh` for headers.
- Use `logic`, `always_ff`, and `always_comb` where applicable.
- Use active-low reset names ending in `_ni`; input names end in `_i`, output names in `_o`.
- Keep one primary module per RTL file and match the file name to the module name.
- Put `` `default_nettype none`` at the start of RTL/testbench source and restore it at the end.
- Document interface behavior before implementing a new block.
- Every functional RTL change needs a self-checking testbench or a clearly documented verification follow-up.
- Do not invent or copy interfaces from course code. Derive them from the project's own documented requirements.

## Shared State Across Codex Chats

- Treat `docs/STATUS.md` as the shared handoff between conversations.
- Update it after completing a milestone, changing a major decision, or obtaining a VCS result.
- Keep status factual and concise: completed work, current blocker, verification evidence, and next step.
