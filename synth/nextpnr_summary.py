#!/usr/bin/env python3
"""Summarize a nextpnr-ecp5 log: final utilisation, Fmax and the critical path.

The critical path is printed one line per run of consecutive cells that belong
to the same RTL signal, so the modules and signals on the path are readable.
Usage: nextpnr_summary.py build/synth/nextpnr.log
"""
import re
import sys

RESOURCES = ("TRELLIS_COMB", "TRELLIS_FF", "DP16KD", "MULT18X18D")


def signal_of(cell):
    """u_conv.u_rq.prod_c_MULT18X18D_P1.P3 -> ('u_conv.u_rq', 'prod_c').

    Yosys and nextpnr append cell-type tags to the RTL signal name when they
    build cells; everything from the first tag on is dropped.
    """
    cell = cell.rsplit(".", 1)[0] if re.search(r"\.[A-Z0-9]+$", cell) else cell
    path, _, leaf = cell.rpartition(".")
    stem = leaf
    while True:
        m = re.search(r"_(TRELLIS|LUT4|CCU2C|PFUMX|L6MUX21|MULT18X18D|DP16KD|DPR16X4|ALU54B)", stem)
        if not m:
            break
        stem = stem[: m.start()]
    stem = stem.split("$")[0]
    return path or "(top)", stem or leaf


def main():
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    text = [re.sub(r"^Info: ?", "", l) for l in lines]

    # final device utilisation block (the last one in the log)
    util_start = max((i for i, l in enumerate(text) if l.startswith("Device utilisation")), default=None)
    print("Resources (LFE5U-85F):")
    if util_start is not None:
        for l in text[util_start + 1 : util_start + 40]:
            name = l.strip().split(":")[0]
            if name in RESOURCES:
                print("  " + l.strip())
    for l in text:
        if l.startswith("ERROR"):
            print(l)

    fmax = [l for l in text if l.startswith("Max frequency for clock")]
    if fmax:
        print(fmax[-1])

    starts = [i for i, l in enumerate(text) if l.startswith("Critical path report for clock")]
    if not starts:
        return
    i = starts[-1]
    print()
    print("Critical path (one line per run of the same signal; segment delay, arrival time):")
    steps = []    # [label, total]
    start_cell = end_cell = split = ""
    cur_cell = ""
    for l in text[i + 2 :]:
        if re.match(r"^[\d.]+ ns logic, [\d.]+ ns routing", l.strip()):
            split = l.strip()
            break
        m = re.match(r"^\s*([\d.]+)\s+([\d.]+)\s+Source\s+(\S+)", l)
        if m:
            cur_cell = m.group(3)
            start_cell = start_cell or cur_cell
            steps.append(["%s: %s" % signal_of(cur_cell), float(m.group(2))])
            continue
        m = re.match(r"^\s*([\d.]+)\s+([\d.]+)\s+Net\s+(\S+)", l)
        if m and steps:
            net = m.group(3)
            # a net that kept its RTL name is a better label than an ABC cell name
            if not net.startswith("$") and "$abc$" not in net:
                steps[-1][0] = "%s: %s" % signal_of(net.split("[")[0])
            steps[-1][1] = float(m.group(2))
            continue
        m = re.match(r"^\s*([\d.]+)\s+([\d.]+)\s+Setup\s+(\S+)", l)
        if m:
            end_cell = m.group(3)
            if steps:
                steps[-1][1] = float(m.group(2))
    groups = []   # [label, cells, arrival]
    for label, total in steps:
        if groups and groups[-1][0] == label:
            groups[-1][1] += 1
            groups[-1][2] = total
        else:
            groups.append([label, 1, total])
    print(f"  from {start_cell}")
    prev = 0.0
    for label, cells, arrival in groups:
        print(f"  {arrival - prev:6.2f} {arrival:7.2f} ns  {label} ({cells} cell{'s' if cells > 1 else ''})")
        prev = arrival
    print(f"  to   {end_cell}")
    if split:
        print(f"  {split}")


if __name__ == "__main__":
    main()
