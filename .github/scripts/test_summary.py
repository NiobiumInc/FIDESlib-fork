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


ICON_OK, ICON_WARN, ICON_BAD = "✅", "⚠️", "❌"


def bar(part, whole, width=10):
    """A text bar for part/whole, so a row shows how much of its suite ran at a glance."""
    if whole == 0:
        return "▱" * width
    filled = round(width * part / whole)
    return "▰" * filled + "▱" * (width - filled)


def percent(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="Test results")
    parser.add_argument("--row", action="append", default=[], metavar="LABEL=GLOB")
    args = parser.parse_args()

    rows = []  # (label, found, total, ran, passed, failed, skipped, disabled, reasons)
    missing = 0
    for row in args.row:
        label, _, pattern = row.partition("=")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            rows.append((label, False, 0, 0, 0, 0, 0, 0, {}))
            missing += 1
            continue
        total, failed, skipped, disabled, reasons = read(files)
        ran = total - skipped - disabled
        rows.append((label, True, total, ran, ran - failed, failed, skipped, disabled, reasons))

    ran = sum(r[3] for r in rows)
    passed = sum(r[4] for r in rows)
    failed = sum(r[5] for r in rows)
    skipped = sum(r[6] for r in rows)
    disabled = sum(r[7] for r in rows)
    if missing or failed:
        headline = f"{ICON_BAD} {failed} failed" if failed else f"{ICON_BAD} {missing} without a result file"
    elif skipped or disabled:
        headline = f"{ICON_WARN} {skipped} skipped" + (f", {disabled} disabled" if disabled else "")
    else:
        headline = f"{ICON_OK} all passed"

    lines = [f"## {args.title}", "", f"**{headline}** · {passed:,} passed of {ran:,} run", "",
             "| | Run | Tests | Ran | Passed | Failed | Skipped | Disabled |", "|:-:|---|---:|---|---:|---:|---:|---:|"]
    for label, found, total, r_ran, r_passed, r_failed, r_skipped, r_disabled, _ in rows:
        if not found:
            lines.append(f"| {ICON_BAD} | {label} | no result file | | | | | |")
            continue
        icon = ICON_BAD if r_failed else (ICON_WARN if r_skipped or r_disabled else ICON_OK)
        lines.append(f"| {icon} | {label} | {total:,} | `{bar(r_ran, total)}` {r_ran:,} ({percent(r_ran, total)}) | {r_passed:,} | {r_failed or '0'} | {r_skipped or '0'} | {r_disabled or '0'} |")
    if len(rows) > 1:
        total_all = sum(r[2] for r in rows)
        lines.append(f"| | **Total** | **{total_all:,}** | **{ran:,}** | **{passed:,}** | **{failed}** | **{skipped}** | **{disabled}** |")

    for label, found, _, _, _, _, r_skipped, _, reasons in rows:
        if not found or not reasons:
            continue
        lines += ["", f"<details><summary>⏭️ <b>{r_skipped} skipped in {label}</b>: why</summary>", "", "| Reason | Tests |", "|---|---:|"]
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
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
