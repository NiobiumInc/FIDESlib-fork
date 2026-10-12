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
    """Counts, skip reasons and per-test times over the reports in `files`."""
    total = failed = skipped = disabled = 0
    reasons = collections.defaultdict(list)
    times = []  # (seconds, test) of every test that ran
    for path in files:
        root = ET.parse(path).getroot()
        for case in root.iter("testcase"):
            total += 1
            name = f"{case.get('classname')}.{case.get('name')}"
            if case.find("skipped") is None and case.get("status") != "notrun":
                times.append((float(case.get("time") or 0), name))
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
    return total, failed, skipped, disabled, reasons, times


def percent(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def duration(seconds):
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def status(found, failed, skipped, disabled):
    if not found:
        return "No result file"
    if failed:
        return "Failed"
    if skipped or disabled:
        return "Passed, with tests not executed"
    return "Passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="Test results")
    parser.add_argument("--row", action="append", default=[], metavar="LABEL=GLOB")
    args = parser.parse_args()

    rows = []  # (label, found, total, executed, passed, failed, skipped, disabled, reasons, times)
    missing = 0
    for row in args.row:
        label, _, pattern = row.partition("=")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            rows.append((label, False, 0, 0, 0, 0, 0, 0, {}, []))
            missing += 1
            continue
        total, failed, skipped, disabled, reasons, times = read(files)
        executed = total - skipped - disabled
        rows.append((label, True, total, executed, executed - failed, failed, skipped, disabled, reasons, times))

    lines = [f"### {args.title}", "",
             "| Run | Status | Tests | Executed | Passed | Failed | Skipped | Disabled | Test time |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, found, total, executed, passed, failed, skipped, disabled, _, times in rows:
        if not found:
            lines.append(f"| {label} | {status(False, 0, 0, 0)} | | | | | | | |")
            continue
        lines.append(f"| {label} | {status(True, failed, skipped, disabled)} | {total:,} | {executed:,} ({percent(executed, total)}) | {passed:,} | {failed} | {skipped} | {disabled} | {duration(sum(t for t, _ in times))} |")
    if len(rows) > 1:
        t = [sum(r[k] for r in rows) for k in range(2, 8)]
        lines.append(f"| **Total** | | **{t[0]:,}** | **{t[1]:,} ({percent(t[1], t[0])})** | **{t[2]:,}** | **{t[3]}** | **{t[4]}** | **{t[5]}** | **{duration(sum(x for r in rows for x, _ in r[9]))}** |")
    lines += ["", "Test time is the sum of the time of every executed test. Runs split into shards or processes that run side by side take less wall-clock time than that sum."]

    slowest = sorted(((x, label, name) for label, found, _, _, _, _, _, _, _, times in rows if found for x, name in times), reverse=True)[:10]
    if slowest:
        lines += ["", "#### Slowest tests", "", "| Test | Run | Time |", "|---|---|---:|"]
        lines += [f"| `{name}` | {label} | {duration(x)} |" for x, label, name in slowest]

    skipped_rows = [(label, reasons) for label, found, _, _, _, _, _, _, reasons, _ in rows if found and reasons]
    if skipped_rows:
        lines += ["", "#### Tests not executed, by reason", "", "| Run | Reason | Tests |", "|---|---|---:|"]
        for label, reasons in skipped_rows:
            for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"| {label} | {reason.replace('|', '/')} | {len(names)} |")

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
