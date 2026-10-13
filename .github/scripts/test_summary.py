#!/usr/bin/env python3
"""Writes a Markdown report of gtest results to the job summary.

    test_summary.py [--title TITLE] [--strict] --row "LABEL=GLOB" [--row "LABEL=GLOB" ...]

Each row sums every gtest XML report the glob matches (the shards of one run, say). A
not-applicable.json next to a report (written by run_gtest.py) lists the tests left out of that
run because they do not apply to the backend, each with its reason. A known-gaps.json next to it
lists the tests that should pass on the backend and do not yet: they are counted apart, as known gaps,
so that a missing feature is not taken for a test that has no meaning on the backend. The report has
a table with, per row, the tests the binary defines, how many were executed, passed, failed, skipped,
not applicable, known gaps and disabled and the time of the tests, a table of the slowest tests, and
a table of the reasons of the tests that were not executed. The Markdown goes to $GITHUB_STEP_SUMMARY when it
is set, and to standard output otherwise.

Exits 1 if a row matched no report, so the job that calls it cannot go green on a missing result.
A failed test is not an error here: the job that ran the tests has already failed. With --strict,
a test that was skipped or disabled is an error too: a test that does not apply to a backend
belongs in that backend's not-applicable list, with its reason, and not in a skip.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


def read(files):
    """Counts, reasons and per-test times over the reports in `files`."""
    total = failed = skipped = disabled = 0
    skip_reasons = collections.defaultdict(list)
    times = []  # (seconds, test) of every test that ran
    not_applicable = {}  # test -> reason, over the not-applicable.json next to each report
    gaps = {}  # test -> reason, over the known-gaps.json next to each report
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
                skip_reasons[message[:140]].append(name)
            elif case.get("status") == "notrun":
                disabled += 1
            else:
                times.append((float(case.get("time") or 0), name))
                if case.find("failure") is not None or case.find("error") is not None:
                    failed += 1
        for file_name, into in (("not-applicable.json", not_applicable), ("known-gaps.json", gaps)):
            extra = os.path.join(os.path.dirname(path), file_name)
            if os.path.exists(extra):
                with open(extra) as f:
                    for item in json.load(f)["tests"]:
                        into[item["test"]] = item["reason"]
    reasons = {}
    for key, found in (("na_reasons", not_applicable), ("gap_reasons", gaps)):
        reasons[key] = collections.defaultdict(list)
        for name, reason in found.items():
            reasons[key][reason].append(name)
    return {"total": total + len(not_applicable) + len(gaps), "failed": failed, "skipped": skipped, "disabled": disabled,
            "not_applicable": len(not_applicable), "known_gaps": len(gaps), "skip_reasons": skip_reasons,
            "times": times, **reasons}


def percent(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def duration(seconds):
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def status(r):
    if r is None:
        return "No result file"
    if r["failed"]:
        return "Failed"
    if r["skipped"] or r["disabled"]:
        return "Passed, with tests skipped or disabled"
    return "Passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="Test results")
    parser.add_argument("--row", action="append", default=[], metavar="LABEL=GLOB")
    parser.add_argument("--strict", action="store_true", help="fail on a skipped or disabled test")
    args = parser.parse_args()

    rows = []  # (label, result or None)
    for row in args.row:
        label, _, pattern = row.partition("=")
        files = sorted(glob.glob(pattern, recursive=True))
        rows.append((label, read(files) if files else None))
    missing = sum(1 for _, r in rows if r is None)

    def executed(r):
        return r["total"] - r["skipped"] - r["disabled"] - r["not_applicable"] - r["known_gaps"]

    lines = [f"### {args.title}", "",
             "| Run | Status | Tests | Executed | Passed | Failed | Skipped | Not applicable | Known gaps | Disabled | Test time |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in rows:
        if r is None:
            lines.append(f"| {label} | {status(None)} | | | | | | | | | |")
            continue
        ran = executed(r)
        lines.append(f"| {label} | {status(r)} | {r['total']:,} | {ran:,} ({percent(ran, r['total'])}) | {ran - r['failed']:,} | {r['failed']} | "
                     f"{r['skipped']} | {r['not_applicable']} | {r['known_gaps']} | {r['disabled']} | {duration(sum(t for t, _ in r['times']))} |")
    found = [r for _, r in rows if r is not None]
    if len(rows) > 1 and found:
        total = sum(r["total"] for r in found)
        ran = sum(executed(r) for r in found)
        lines.append(f"| **Total** | | **{total:,}** | **{ran:,} ({percent(ran, total)})** | **{ran - sum(r['failed'] for r in found):,}** | "
                     f"**{sum(r['failed'] for r in found)}** | **{sum(r['skipped'] for r in found)}** | **{sum(r['not_applicable'] for r in found)}** | "
                     f"**{sum(r['known_gaps'] for r in found)}** | **{sum(r['disabled'] for r in found)}** | **{duration(sum(t for r in found for t, _ in r['times']))}** |")
    lines += ["", "Test time is the sum of the time of every executed test. Runs split into shards or processes that run side by side take less wall-clock time than that sum."]

    slowest = sorted(((t, label, name) for label, r in rows if r for t, name in r["times"]), reverse=True)[:10]
    if slowest:
        lines += ["", "#### Slowest tests", "", "| Test | Run | Time |", "|---|---|---:|"]
        lines += [f"| `{name}` | {label} | {duration(t)} |" for t, label, name in slowest]

    for heading, key in (("Tests not applicable, by reason", "na_reasons"), ("Known gaps, by reason", "gap_reasons"),
                         ("Tests skipped, by reason", "skip_reasons")):
        section = [(label, reasons) for label, r in rows if r for reasons in [r[key]] if reasons]
        if section:
            lines += ["", f"#### {heading}", "", "| Run | Reason | Tests |", "|---|---|---:|"]
            for label, reasons in section:
                for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                    lines.append(f"| {label} | {reason.replace('|', '/')} | {len(names)} |")

    text = "\n".join(lines) + "\n"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    strict_failures = sum(r["skipped"] + r["disabled"] for r in found) if args.strict else 0
    if strict_failures:
        print(f"::error::{strict_failures} test(s) were skipped or disabled; a test that does not apply goes in the not-applicable list with its reason.")
    return 1 if (missing or strict_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
