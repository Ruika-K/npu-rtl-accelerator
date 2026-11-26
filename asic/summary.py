#!/usr/bin/env python3
"""Summarize a LibreLane run: area, cells, timing per corner, power, DRC / LVS / antenna.

    asic/summary.py <run directory> [--markdown]

Reads <run>/final/metrics.json. Fmax is estimated from the worst setup slack over
all corners: 1 / (clock period - slack). Exits non-zero if DRC, LVS or setup /
hold timing is not clean.
"""
import json
import os
import sys


def main():
    run = sys.argv[1]
    md = "--markdown" in sys.argv
    m = json.load(open(os.path.join(run, "final", "metrics.json")))
    cfg = json.load(open(os.path.join(run, "resolved.json")))
    period = float(cfg["CLOCK_PERIOD"])

    def get(key, default=None):
        return m.get(key, default)

    setup = get("timing__setup__ws")
    hold = get("timing__hold__ws")
    rows = [
        ("Design", cfg["DESIGN_NAME"]),
        ("PDK / library", "%s / %s" % (cfg.get("PDK"), cfg.get("STD_CELL_LIBRARY"))),
        ("Die area", "%.3f mm²" % (get("design__die__area", 0) / 1e6)),
        ("Core utilization", "%.1f %%" % (100 * get("design__instance__utilization", 0))),
        ("Standard cells", "%s (area %.3f mm²)" % (get("design__instance__count__stdcell"),
                                                    get("design__instance__area__stdcell", 0) / 1e6)),
        ("Clock period", "%.2f ns (%.1f MHz target)" % (period, 1000 / period)),
        ("Worst setup slack (all corners)", "%.3f ns" % setup if setup is not None else "n/a"),
        ("Worst hold slack (all corners)", "%.3f ns" % hold if hold is not None else "n/a"),
        ("Estimated Fmax (worst corner)", "%.1f MHz" % (1000 / (period - setup)) if setup is not None else "n/a"),
        ("Setup slack, typical corner (nom_tt_025C_1v80)", "%s ns" % get("timing__setup__ws__corner:nom_tt_025C_1v80")),
        ("Total power (typical corner)", "%.2f mW" % (1000 * get("power__total", 0))),
        ("Routing DRC errors", get("route__drc_errors")),
        ("Magic DRC errors", get("magic__drc_error__count")),
        ("KLayout DRC errors", get("klayout__drc_error__count")),
        ("LVS errors", get("design__lvs_error__count")),
        ("Antenna violating nets", get("antenna__violating__nets")),
        ("Max slew / cap / fanout violations", "%s / %s / %s" % (get("design__max_slew_violation__count"),
                                                                 get("design__max_cap_violation__count"),
                                                                 get("design__max_fanout_violation__count"))),
    ]
    if md:
        print("| Metric | Value |\n| --- | --- |")
        for k, v in rows:
            print("| %s | %s |" % (k, v))
    else:
        for k, v in rows:
            print("%-48s %s" % (k, v))
    bad = [k for k in ("route__drc_errors", "magic__drc_error__count", "klayout__drc_error__count",
                       "design__lvs_error__count") if get(k) not in (0, None)]
    if setup is not None and setup < 0:
        bad.append("setup slack")
    if hold is not None and hold < 0:
        bad.append("hold slack")
    if bad:
        print("NOT CLEAN: " + ", ".join(bad))
        sys.exit(1)
    print("CLEAN: DRC, LVS, setup and hold")


if __name__ == "__main__":
    main()
