#!/usr/bin/env python3
"""Merge the cover-property counts of all Verilator runs and check closure.

Each testbench run writes a coverage database (+cov=<file>, see
tb/common/verilator_main.cpp). This script sums the hits of every cover
property of tb/sva over all instances and all runs, prints one line per
property with the tests that reached it, and exits with an error if a
property was never reached, unless tb/sva/cover_waivers.txt lists it with a
reason.

Usage: cover_report.py build/cov/*.dat [--waivers FILE] [--markdown]
"""
import os
import re
import sys


def parse(path):
    """Yield (checker, name, count) for every user cover point in one database."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("C '"):
                continue
            body, _, count = line.rstrip().rpartition("' ")
            fields = dict(kv.split("\x02", 1) for kv in body[3:].split("\x01") if "\x02" in kv)
            page = fields.get("page", "")
            if not page.startswith("v_user/"):
                continue
            # Verilator appends the parameter values to the module name
            checker = page[len("v_user/"):].split("__", 1)[0]
            yield checker, fields.get("o", "?"), int(count)


def load_waivers(path):
    waivers = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, _, reason = line.partition(":")
            waivers[key.strip()] = reason.strip()
    return waivers


def main():
    args = sys.argv[1:]
    markdown = "--markdown" in args
    waiver_file = None
    if "--waivers" in args:
        waiver_file = args[args.index("--waivers") + 1]
    files = [a for a in args if a.endswith(".dat")]
    if not files:
        sys.exit("cover_report.py: no coverage databases given")
    waivers = load_waivers(waiver_file)

    total = {}    # (checker, name) -> hits
    tests = {}    # (checker, name) -> tests that reached it
    for path in sorted(files):
        test = re.sub(r"\.dat$", "", os.path.basename(path))
        for checker, name, count in parse(path):
            key = (checker, name)
            total[key] = total.get(key, 0) + count
            tests.setdefault(key, [])
            if count and test not in tests[key]:
                tests[key].append(test)

    missed, waived = [], []
    if markdown:
        print("| Checker | Cover property | Hits | Reached by |")
        print("| --- | --- | ---: | --- |")
    for key in sorted(total):
        checker, name = key
        hits = total[key]
        where = ", ".join(tests[key]) if tests[key] else "-"
        full = f"{checker}.{name}"
        if hits == 0:
            (waived if full in waivers else missed).append(full)
            if full in waivers:
                where = "waived: " + waivers[full]
        if markdown:
            print(f"| `{checker}` | `{name}` | {hits} | {where} |")
        else:
            print(f"{checker:22s} {name:20s} {hits:10d}  {where}")

    reached = sum(1 for k in total if total[k])
    print()
    print(f"cover properties: {len(total)}, reached: {reached}, waived: {len(waived)}, missed: {len(missed)}"
          f" ({len(files)} runs)")
    stale = sorted(w for w in waivers if tuple(w.split(".", 1)) not in total or total[tuple(w.split(".", 1))])
    for w in stale:
        print(f"stale waiver (property reached or gone): {w}")
    for m in missed:
        print(f"MISSED: {m}")
    if missed or stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
