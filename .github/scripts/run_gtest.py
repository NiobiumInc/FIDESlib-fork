#!/usr/bin/env python3
"""Runs a gtest binary without the tests that do not apply to a backend, and records which they were.

    run_gtest.py --binary BINARY --list FILE --xml OUT.xml [--dry-run] [-- GTEST_ARGS...]

FILE has one entry per line, `PATTERN | REASON`: PATTERN is one gtest pattern (or several joined
with `:`) and REASON says why those tests make no sense on the backend. Blank lines and lines that
start with `#` are ignored. The tests the patterns match are left out of the run with a negative
gtest filter and written to not-applicable.json next to the XML report, so that the test report
counts them as not applicable and not as skipped.

A pattern that matches no test is an error: it is a line left behind by a test that was renamed
or removed. The JSON is written by the first shard only (GTEST_SHARD_INDEX 0, or no sharding),
because every shard excludes the same tests. Exit status: that of the binary, 2 for a stale
pattern or a malformed list.
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys


def read_list(path):
    entries = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            pattern, bar, reason = text.partition("|")
            if not bar or not pattern.strip() or not reason.strip():
                sys.exit(f"{path}:{number}: expected `PATTERN | REASON`, got: {text}")
            entries.append((pattern.strip(), reason.strip()))
    return entries


def list_tests(binary):
    out = subprocess.run([binary, "--gtest_list_tests"], check=True, capture_output=True, text=True).stdout
    names, suite = [], ""
    for line in out.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            suite = line.split("#")[0].strip()
        else:
            names.append(suite + line.split("#")[0].strip())
    return names


def matches(name, pattern):
    return any(fnmatch.fnmatchcase(name, p) for p in pattern.split(":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--dry-run", action="store_true", help="check the list and print what it leaves out; do not run the binary")
    parser.add_argument("gtest_args", nargs="*")
    args = parser.parse_args()

    entries = read_list(args.list)
    names = list_tests(args.binary)
    excluded, stale = [], []
    for pattern, reason in entries:
        hit = [n for n in names if matches(n, pattern)]
        if not hit:
            stale.append(pattern)
        excluded += [{"test": n, "reason": reason} for n in hit]
    if stale:
        sys.exit("These patterns match no test (renamed or removed?): " + ", ".join(stale))
    # A test that two entries match is listed once, under the first.
    seen, unique = set(), []
    for item in excluded:
        if item["test"] not in seen:
            seen.add(item["test"])
            unique.append(item)

    if args.dry_run:
        by_reason = {}
        for item in unique:
            by_reason.setdefault(item["reason"], []).append(item["test"])
        print(f"{len(unique)} of {len(names)} tests do not apply:")
        for reason, tests in by_reason.items():
            print(f"  {len(tests):4}  {reason}")
        return 0

    out_dir = os.path.dirname(os.path.abspath(args.xml))
    os.makedirs(out_dir, exist_ok=True)
    if os.environ.get("GTEST_SHARD_INDEX", "0") == "0":
        with open(os.path.join(out_dir, "not-applicable.json"), "w") as f:
            json.dump({"list": args.list, "tests": unique}, f, indent=1)

    cmd = [args.binary, f"--gtest_output=xml:{args.xml}"]
    if unique:
        cmd.append("--gtest_filter=-" + ":".join(sorted(seen)))
    cmd += args.gtest_args
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
