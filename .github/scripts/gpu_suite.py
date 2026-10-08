#!/usr/bin/env python3
"""Runs the FIDESlib GPU test suite and writes a report of what failed.

Three passes, each with its own gtest XML report:

1. Gate: every test except those in benchmarks.txt, known-failures.txt and
   flaky.txt. A failing test gets up to RETRIES more attempts, because OpenFHE
   seeds its random generator on every run and a few tests flip with no source
   change. A test that fails every attempt fails the check.
2. Flaky: the tests in flaky.txt, as a report that never fails the check.
3. Known failures, with --known-failures: the tests in known-failures.txt, as a
   report that never fails the check.

With --only, the gate runs just the tests that match those gtest patterns (still
minus the lists), so one pass can cover a part of the suite that needs its own
environment, such as the concurrent-operations mode. --title names the summary
section, so two passes in one job stay apart.

With --isolate, every test runs in a process of its own. A CUDA error such as an
illegal memory access leaves the context unusable for the rest of the process, so
every test after it in the same process fails too, and a CudaError thrown where
nothing catches it ends the process. Isolated, the test that hit it fails alone,
and a test that ends without writing its report counts as a failure, never as a
pass.

With --repeat N, the gate runs its tests N times over in one process
(--gtest_repeat), and a test that fails in any repetition fails. gtest's XML
report keeps only the last repetition, so the failures of the earlier ones are
read from the log. --retries sets the extra attempts (default RETRIES); 0 makes
a failure final, for a pass whose failures a retry in a fresh process would hide.

The suite prints too much for a step log, so each pass writes its full output
to <out>/<pass>.log and the step log keeps only gtest's progress lines. The
report is Markdown, appended to $GITHUB_STEP_SUMMARY when that is set and
printed otherwise. Exit status: 0 when no gate test failed every attempt, 1
when one did, and the suite's own status when it crashed before writing its
report.
"""

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

LISTS = ("benchmarks.txt", "known-failures.txt", "flaky.txt")

# Extra attempts for a gate test that fails.
RETRIES = 2


def read_patterns(path):
    """Returns the gtest filter patterns in a list file, without comments."""
    patterns = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                patterns.append(line)
    return patterns


def summarize(message, location=True):
    """Shortens a gtest message to one table cell, with or without its file:line."""
    lines = [line.strip() for line in message.strip().splitlines() if line.strip()]
    where = ""
    if lines and re.search(r":\d+$", lines[0]):
        where = os.path.basename(lines.pop(0))
    text = " ".join(lines)
    if len(text) > 220:
        text = text[:217] + "..."
    text = f"{where}: {text}" if where and location else text
    return text.replace("|", "\\|")


def parse(report):
    """Returns {test: (status, message)}, status being passed, failed, skipped or disabled.

    A disabled test is one whose name starts with DISABLED_: gtest lists it in
    the report as suppressed without running it. Its message is its file:line.
    """
    results = {}
    for suite in ET.parse(report).getroot().iter("testsuite"):
        for case in suite.iter("testcase"):
            name = f"{suite.get('name')}.{case.get('name')}"
            failure = case.find("failure")
            if case.get("result") == "suppressed":
                where = os.path.basename(case.get("file") or "")
                results[name] = ("disabled", f"{where}:{case.get('line')}" if where else "")
            elif failure is not None:
                text = failure.get("message") or failure.text or ""
                results[name] = ("failed", summarize(text))
            elif case.find("skipped") is not None or case.get("result") == "skipped":
                skipped = case.find("skipped")
                text = "" if skipped is None else (skipped.get("message") or skipped.text or "")
                results[name] = ("skipped", summarize(text, location=False))
            else:
                results[name] = ("passed", "")
    return results


# gtest's line for a test that failed, with or without a ", where GetParam() = ..." part.
FAILED_LINE = re.compile(r"^\[  FAILED  \] ([^\s,]+)(?:, where .*)? \(\d+ ms\)$")


def run(binary, gtest_filter, report, repeat=1):
    """Runs the suite. Returns (exit status, results), results None on no report.

    The full output goes to the report's .log twin; only gtest's own lines,
    which start with '[', reach stdout. With repeat > 1 the tests run that many
    times over in this one process, and a test that failed in any repetition is
    failed in the results.
    """
    if os.path.exists(report):
        os.remove(report)
    cmd = [binary, f"--gtest_filter={gtest_filter}", f"--gtest_output=xml:{report}"]
    if repeat > 1:
        cmd.append(f"--gtest_repeat={repeat}")
    log_path = os.path.splitext(report)[0] + ".log"
    print("$ " + " ".join(cmd), flush=True)
    print(f"Full output: {os.path.basename(log_path)} in the test reports artifact", flush=True)
    failed_in_log = set()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        for line in proc.stdout:
            log.write(line)
            if line.startswith("["):
                sys.stdout.write(line)
                sys.stdout.flush()
                match = FAILED_LINE.match(line.rstrip("\n"))
                if match:
                    failed_in_log.add(match.group(1))
        status = proc.wait()
    if not os.path.isfile(report) or os.path.getsize(report) == 0:
        return status, None
    results = parse(report)
    if repeat > 1:
        # The report holds the last repetition only.
        for name in failed_in_log:
            if results.get(name, ("", ""))[0] != "failed":
                results[name] = ("failed", f"failed in one of {repeat} repetitions; see the log")
    return status, results


def list_tests(binary, gtest_filter):
    """Returns the full names of the tests a filter selects, in gtest's order."""
    listing = subprocess.run([binary, "--gtest_list_tests", f"--gtest_filter={gtest_filter}"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    names, suite = [], None
    for line in listing.stdout.splitlines():
        text = line.split("#", 1)[0].rstrip()
        if not text:
            continue
        if not line.startswith(" "):
            suite = text if text.endswith(".") else None
        elif suite:
            names.append(suite + text.strip())
    return names


def run_isolated(binary, names, out_dir):
    """Runs each test in a process of its own. Returns (exit status, results).

    A test whose process ends without its report, or without the test in it, is
    recorded as failed with how the process ended.
    """
    os.makedirs(out_dir, exist_ok=True)
    status, results = 0, {}
    for index, name in enumerate(names):
        code, found = run(binary, name, os.path.join(out_dir, f"{index:03d}.xml"))
        if found is None or name not in found:
            how = f"signal {-code}" if code < 0 else f"exit {code}"
            results[name] = ("failed", f"crashed: the process ended ({how}) without reporting this test")
            print(f"::warning::{name} ended its process ({how}) without reporting a result", flush=True)
            status = status or 1
        else:
            results.update(found)
            status = status or code
    return status, results


def counts(results):
    by = {"passed": 0, "failed": 0, "skipped": 0, "disabled": 0}
    for status, _ in results.values():
        by[status] += 1
    return len(results), by["passed"], by["failed"], by["skipped"], by["disabled"]


def count_row(label, results, status):
    if results is None:
        return f"| {label} | crashed (exit {status}, no report) | | | | |"
    total, passed, failed, skipped, disabled = counts(results)
    return f"| {label} | {total} | {passed} | {failed} | {skipped} | {disabled} |"


def report_only(binary, patterns, report, isolate=False):
    """Runs a report-only pass. Returns (exit status, results) or None if empty."""
    if not patterns:
        return None
    if isolate:
        return run_isolated(binary, list_tests(binary, ":".join(patterns)), os.path.splitext(report)[0])
    return run(binary, ":".join(patterns), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", help="the fideslib-test executable")
    parser.add_argument("lists", help="directory holding " + ", ".join(LISTS))
    parser.add_argument("out", help="directory for the gtest XML reports")
    parser.add_argument("--known-failures", action="store_true",
                        help="also run known-failures.txt, as a report")
    parser.add_argument("--only", default="*",
                        help="gtest patterns the gate runs, ':'-separated (default: every test)")
    parser.add_argument("--title", default="GPU test suite",
                        help="heading of this pass's section in the job summary")
    parser.add_argument("--isolate", action="store_true",
                        help="run every test in a process of its own")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the gate tests this many times over in one process")
    parser.add_argument("--retries", type=int, default=RETRIES,
                        help=f"extra attempts for a gate test that fails (default {RETRIES})")
    args = parser.parse_args()
    if args.repeat < 1 or args.retries < 0:
        parser.error("--repeat must be at least 1 and --retries at least 0")
    if args.repeat > 1 and args.isolate:
        parser.error("--repeat runs the tests together in one process; it cannot go with --isolate")

    os.makedirs(args.out, exist_ok=True)
    benchmarks, known, flaky = (read_patterns(os.path.join(args.lists, name)) for name in LISTS)

    # 1. Gate, with up to args.retries more attempts for whatever fails.
    excluded = benchmarks + known + flaky
    gate_filter = args.only + ("-" + ":".join(excluded) if excluded else "")
    if args.isolate:
        gate_status, gate = run_isolated(args.binary, list_tests(args.binary, gate_filter),
                                         os.path.join(args.out, "gate"))
    else:
        gate_status, gate = run(args.binary, gate_filter, os.path.join(args.out, "gate.xml"), args.repeat)
    exit_status = 0
    failed = []        # gate tests that failed their first attempt
    pending = []       # gate tests that have failed every attempt so far
    attempts = {}      # test -> outcome of each retry
    last_message = {}  # test -> message of its latest failure
    retry_rows = []    # (label, results, status) of each retry, for the counts table
    if gate is None:
        print(f"::error::The GPU suite exited {gate_status} without writing its report; it probably crashed.")
        exit_status = gate_status or 1
    else:
        failed = [name for name, (status, _) in gate.items() if status == "failed"]
        last_message = {name: gate[name][1] for name in failed}
        pending = list(failed)
        for attempt in range(1, args.retries + 1):
            if not pending:
                break
            print(f"Retry {attempt} of {args.retries}: " + " ".join(pending), flush=True)
            if args.isolate:
                status, results = run_isolated(args.binary, pending,
                                               os.path.join(args.out, f"gate-retry{attempt}"))
            else:
                status, results = run(args.binary, ":".join(pending),
                                      os.path.join(args.out, f"gate-retry{attempt}.xml"))
            retry_rows.append((f"Check, retry {attempt}", results, status))
            still = []
            for name in pending:
                if results is None:
                    outcome, message = "crashed", ""
                else:
                    outcome, message = results.get(name, ("not run", ""))
                attempts.setdefault(name, []).append(outcome)
                if outcome == "failed":
                    last_message[name] = message
                if outcome != "passed":
                    still.append(name)
            pending = still
        if not failed and gate_status != 0:
            print(f"::error::The GPU suite exited {gate_status} but its report lists no failing test.")
            exit_status = gate_status

    recovered = [n for n in failed if n not in pending]
    for name in recovered:
        print(f"::warning::Passed on retry, possibly flaky: {name}")
    if pending:
        print(f"::error::Failed all {args.retries + 1} attempt(s): " + " ".join(pending))
        exit_status = 1

    # 2 and 3. Report-only passes.
    flaky_pass = report_only(args.binary, flaky, os.path.join(args.out, "flaky.xml"), args.isolate)
    known_pass = (report_only(args.binary, known, os.path.join(args.out, "known-failures.xml"), args.isolate)
                  if args.known_failures else None)

    lines = [f"## {args.title}", ""]
    if exit_status == 0:
        lines.append(f"**Check: passed.** No test outside the lists failed all {args.retries + 1} attempt(s).")
    elif pending:
        lines.append(f"**Check: failed.** {len(pending)} test(s) failed all {args.retries + 1} attempt(s).")
    else:
        lines.append("**Check: failed.** The suite did not finish; see the job log.")
    if args.isolate:
        lines += ["", "Each test ran in a process of its own."]
    if args.repeat > 1:
        lines += ["", f"The tests ran {args.repeat} times over in one process; a failure in any repetition counts."]
    lines += ["", "| Pass | Tests | Passed | Failed | Skipped | Disabled |", "|---|---:|---:|---:|---:|---:|"]
    lines.append(count_row("Check", gate, gate_status))
    for label, results, status in retry_rows:
        lines.append(count_row(label, results, status))
    if flaky_pass:
        lines.append(count_row("Flaky (report only)", flaky_pass[1], flaky_pass[0]))
    if known_pass:
        lines.append(count_row("Known failures (report only)", known_pass[1], known_pass[0]))

    if failed:
        lines += ["", "### Check failures", "", "| Test | Attempts | Message |", "|---|---|---|"]
        for name in failed:
            outcomes = ", ".join(["failed"] + attempts.get(name, []))
            lines.append(f"| `{name}` | {outcomes} | {last_message[name]} |")

    # Tests that call GTEST_SKIP count as passing; list them with their reasons
    # so a skip that hides a known bug stays visible.
    self_skipped = {}
    for name, (status, reason) in (gate or {}).items():
        if status == "skipped":
            self_skipped.setdefault(reason or "no reason given", []).append(name)
    if self_skipped:
        total = sum(len(names) for names in self_skipped.values())
        print(f"::notice::{total} tests skipped themselves and count as passing. See the job summary.")
        lines += ["", "### Skipped by the tests themselves", "",
                  "The check counts these as passing. Each reason is the test's own GTEST_SKIP message.", "",
                  "| Reason | Tests |", "|---|---|"]
        for reason, names in sorted(self_skipped.items(), key=lambda item: -len(item[1])):
            lines.append(f"| {reason} | " + ", ".join(f"`{n}`" for n in sorted(names)) + " |")

    # Tests disabled in the source never run; list them so one that hides a
    # known bug stays visible.
    disabled = sorted((name, where) for name, (status, where) in (gate or {}).items()
                      if status == "disabled")
    if disabled:
        print(f"::notice::{len(disabled)} tests are disabled in the source and did not run. See the job summary.")
        lines += ["", "### Disabled in the source", "",
                  "Their names start with `DISABLED_`, so gtest lists them without running them.", "",
                  "| Test | Location |", "|---|---|"]
        lines += [f"| `{name}` | {where} |" for name, where in disabled]

    if flaky_pass and flaky_pass[1] is not None:
        flaky_failed = [(n, m) for n, (s, m) in flaky_pass[1].items() if s == "failed"]
        lines += ["", "### Flaky tests (report only)", ""]
        if flaky_failed:
            lines += ["| Test | Message |", "|---|---|"]
            lines += [f"| `{n}` | {m} |" for n, m in flaky_failed]
            print(f"::notice::Flaky tests (report only): {len(flaky_failed)} of "
                  f"{len(flaky_pass[1])} failed. See the job summary.")
        else:
            lines.append(f"All {len(flaky_pass[1])} passed in this run.")

    if known_pass and known_pass[1] is not None:
        lines += ["", "### Known failures (report only)", "",
                  "| Test | Result | Message |", "|---|---|---|"]
        for name, (status, message) in sorted(known_pass[1].items()):
            lines.append(f"| `{name}` | {status} | {message} |")

    not_run = []
    if benchmarks:
        not_run.append("- Timing benchmarks: " + ", ".join(f"`{p}`" for p in benchmarks))
    if not args.known_failures and known:
        not_run.append(f"- Known failures: {len(known)} patterns in `known-failures.txt`. "
                       "Run the workflow by hand with **known_failures** to include them.")
    if not_run:
        lines += ["", "### Not run in this job", ""] + not_run

    text = "\n".join(lines) + "\n"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    else:
        print(text)
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
