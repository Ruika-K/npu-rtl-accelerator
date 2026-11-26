#!/usr/bin/env python3
"""Fail unless a cocotb results file exists and reports every test as passed.

cocotb's makefiles exit 0 when tests fail, and write no results when the
simulator stops early (for example on a failed assertion), so make cocotb
checks the file instead of the exit code.
"""
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
try:
    root = ET.parse(path).getroot()
except (OSError, ET.ParseError) as e:
    sys.exit(f"check_results.py: no cocotb results ({e}); the simulation stopped before the tests finished")
cases = root.findall(".//testcase")
bad = [c.get("name") for c in cases if c.find("failure") is not None or c.find("error") is not None]
print(f"cocotb: {len(cases)} tests, {len(cases) - len(bad)} passed")
for name in bad:
    print(f"FAILED: {name}")
sys.exit(1 if bad or not cases else 0)
