#!/usr/bin/env python3
"""Print the parts of a LibreLane sign-off STA that explain a timing result.

    asic/sta_digest.py <run directory>

From the last OpenROAD.STAPostPNR step: the worst setup path of the slow
corner, the worst hold path of every corner that has a hold violation, and the
max slew / cap / fanout violators (first lines), so a CI log alone shows where
the time goes.
"""
import glob
import os
import re
import sys


def first_path(text):
    m = re.search(r"Startpoint:.*?slack \([A-Z]+\)", text, re.S)
    return m.group(0) if m else "(no path)"


def section(text, header, lines=60):
    i = text.find(header)
    if i < 0:
        return "(section not found)"
    return "\n".join(text[i:].splitlines()[:lines])


def main():
    run = sys.argv[1]
    steps = sorted(glob.glob(os.path.join(run, "*-openroad-stapostpnr")),
                   key=lambda p: int(os.path.basename(p).split("-")[0]))
    if not steps:
        sys.exit("no OpenROAD.STAPostPNR step in %s" % run)
    sta = steps[-1]
    corners = sorted(d for d in os.listdir(sta) if os.path.isdir(os.path.join(sta, d)))
    print("STA step:", sta, "corners:", ", ".join(corners))

    def read(corner, name):
        p = os.path.join(sta, corner, name)
        return open(p, errors="replace").read() if os.path.exists(p) else ""

    slow = next((c for c in corners if c.startswith("nom_ss")), corners[0])
    print("\n===== worst setup path, %s =====" % slow)
    print(first_path(read(slow, "max.rpt")))
    for c in corners:
        mn = read(c, "min.rpt")
        m = re.search(r"([-\d.]+)\s+slack \(VIOLATED\)", mn)
        if m:
            print("\n===== worst hold path, %s (slack %s) =====" % (c, m.group(1)))
            print(first_path(mn))
    print("\n===== slew / cap / fanout violators, %s =====" % slow)
    print(section(read(slow, "checks.rpt"), "report_check_types -max_slew -max_cap -max_fanout -violators", 80))


if __name__ == "__main__":
    main()
