#!/usr/bin/env python3
"""Writes a Markdown table of gtest results to the job summary.

    test_summary.py [--title TITLE] --row "LABEL=GLOB" [--row "LABEL=GLOB" ...]

Each row sums every gtest XML report the glob matches (the shards of one run, say). The table
gives, per row, the tests the binary defines, how many ran, passed, failed and were skipped, and
how many are disabled. Tests that skipped themselves are listed by reason under the table, so a
skip is never silent. The Markdown goes to $GITHUB_STEP_SUMMARY when it is set, and to standard
output otherwise. Exits 1 if a row matched no report, so the job that calls it cannot go green on a
missing result. A failed test is not an error here: the job that ran the tests has already failed.
"""
import argparse
import collections
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET


def read(files):
    """Counts and skip reasons over the reports in `files`."""
    total = failed = skipped = disabled = 0
    reasons = collections.defaultdict(list)
    for path in files:
        root = ET.parse(path).getroot()
        for case in root.iter("testcase"):
            total += 1
            name = f"{case.get('classname')}.{case.get('name')}"
            skip = case.find("skipped")
            if skip is not None:
                skipped += 1
                message = re.sub(r"\s+", " ", skip.get("message") or "")
                message = re.sub(r"^.*?/test/\w+\.\w+:\d+ ", "", message) or "no reason recorded"
                reasons[message[:140]].append(name)
            elif case.get("status") == "notrun":
                disabled += 1
            elif case.find("failure") is not None or case.find("error") is not None:
                failed += 1
    return total, failed, skipped, disabled, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="Test results")
    parser.add_argument("--row", action="append", default=[], metavar="LABEL=GLOB")
    args = parser.parse_args()

    lines = [f"## {args.title}", "", "| Run | Tests | Ran | Passed | Failed | Skipped | Disabled |", "|---|---:|---:|---:|---:|---:|---:|"]
    problems = 0
    skip_sections = []
    for row in args.row:
        label, _, pattern = row.partition("=")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            lines.append(f"| {label} | no result file | | | | | |")
            problems += 1
            continue
        total, failed, skipped, disabled, reasons = read(files)
        ran = total - skipped - disabled
        lines.append(f"| {label} | {total} | {ran} | {ran - failed} | {failed} | {skipped} | {disabled} |")
        if reasons:
            skip_sections.append((label, reasons))
    for label, reasons in skip_sections:
        lines += ["", f"<details><summary>Skipped in {label}</summary>", "", "| Reason | Tests |", "|---|---:|"]
        for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {reason.replace('|', '/')} | {len(names)} |")
        lines += ["", "</details>"]
    text = "\n".join(lines) + "\n"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
