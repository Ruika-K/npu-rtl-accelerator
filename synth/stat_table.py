#!/usr/bin/env python3
"""Turn a non-flattened `yosys stat -tech xilinx` report into a Markdown table.

Each row is one module instance path below the top, with the cells of that
module and everything under it (inclusive), so the expensive blocks stand out.
Usage: stat_table.py build/synth/xilinx_stat.txt [top]
"""
import re
import sys
from collections import defaultdict

CLASSES = (
    ("LUT", re.compile(r"^LUT[1-6]$")),
    ("MUXF", re.compile(r"^MUXF[78]$")),
    ("CARRY4", re.compile(r"^CARRY4$")),
    ("FF", re.compile(r"^FD[A-Z]*$")),
    ("DSP48E1", re.compile(r"^DSP48E1$")),
    ("RAMB18", re.compile(r"^RAMB18E1$")),
    ("RAMB36", re.compile(r"^RAMB36E1$")),
    ("LUTRAM", re.compile(r"^RAM(32|64|128|256)[MX0-9A-Z]*$")),
    ("SRL", re.compile(r"^SRL")),
)
COLS = [c for c, _ in CLASSES]


def parse(path):
    """Return {module: {cell_type: count}} for every module section."""
    mods, cur, in_cells = {}, None, False
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^=== (.+) ===\s*$", line)
        if m:
            cur = m.group(1)
            in_cells = False
            if cur != "design hierarchy":
                mods[cur] = {}
            continue
        if cur is None or cur == "design hierarchy":
            continue
        if re.match(r"^\s+Number of cells:", line):
            in_cells = True
            continue
        if in_cells:
            m = re.match(r"^\s+(\S+)\s+(\d+)\s*$", line)
            if m:
                mods[cur][m.group(1)] = int(m.group(2))
            elif line.strip():
                in_cells = False
    return mods


def short(name):
    # $paramod$abc\sram_1r1w\WIDTH=64... -> sram_1r1w
    name = name.lstrip("\\")
    if name.startswith("$paramod"):
        parts = name.split("\\")
        return parts[1] if len(parts) > 1 else name
    return name


def classify(cells):
    out = defaultdict(int)
    for t, n in cells.items():
        for col, rx in CLASSES:
            if rx.match(t):
                out[col] += n
                break
    return out


def main():
    path = sys.argv[1]
    mods = parse(path)
    top = sys.argv[2] if len(sys.argv) > 2 else "npu_top"
    if top not in mods:
        sys.exit(f"stat_table: module {top} not found in {path}")

    memo = {}

    def total(mod):
        if mod in memo:
            return memo[mod]
        acc = classify(mods[mod])
        for t, n in mods[mod].items():
            if t in mods:
                sub = total(t)
                for c in COLS:
                    acc[c] += n * sub[c]
        memo[mod] = acc
        return acc

    rows = []

    def walk(mod, label, depth, count):
        rows.append((label, depth, count, total(mod)))
        for t, n in sorted(mods[mod].items()):
            if t in mods:
                walk(t, short(t), depth + 1, n)

    walk(top, top, 0, 1)
    used = [c for c in COLS if any(r[3][c] for r in rows)]
    print("| module | inst | " + " | ".join(used) + " |")
    print("|---|---:|" + "---:|" * len(used))
    for label, depth, count, t in rows:
        name = "&nbsp;&nbsp;" * depth + label
        print(f"| {name} | {count} | " + " | ".join(str(t[c]) for c in used) + " |")
    print()
    print("Counts are per instance and include submodules; multiply by inst for the total.")


if __name__ == "__main__":
    main()
