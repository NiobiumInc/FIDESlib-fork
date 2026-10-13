#!/usr/bin/env python3
"""Runs a gtest binary without the tests that do not apply to a backend, and records which they were.

    run_gtest.py --binary BINARY --list FILE [--gaps FILE] --xml OUT.xml [--dry-run] [-- GTEST_ARGS...]

FILE has one entry per line, `PATTERN | REASON`: PATTERN is one gtest pattern (or several joined
with `:`) and REASON says why those tests make no sense on the backend. Blank lines and lines that
start with `#` are ignored. The tests the patterns match are left out of the run with a negative
gtest filter and written to not-applicable.json next to the XML report, so that the test report
counts them as not applicable and not as skipped.

--gaps names a second list of the same format: tests that should pass on the backend and do not
yet, because it lacks something they need. They are left out of the run too, but they are written
to known-gaps.json and the report counts them as known gaps, apart from the not-applicable ones, so
that a missing feature is never taken for a test that has no meaning on the backend. A test that
both lists match is not applicable.

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


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def read_list(path):
    entries = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            pattern, bar, reason = text.partition("|")
            if not bar or not pattern.strip() or not reason.strip():
                fail(f"{path}:{number}: expected `PATTERN | REASON`, got: {text}")
            entries.append((pattern.strip(), reason.strip()))
    return entries


def list_tests(binary, env=None):
    """The tests the binary defines, as gtest lists them. `env` adds to the environment of the listing:
    tests are registered per backend (test/TestRegistration.hpp), so the list depends on it."""
    out = subprocess.run([binary, "--gtest_list_tests"], check=True, capture_output=True, text=True,
                         env={**os.environ, **env} if env else None).stdout
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
    parser.add_argument("--gaps", help="list of the tests that should pass on this backend and do not yet")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--dry-run", action="store_true", help="check the list and print what it leaves out; do not run the binary")
    parser.add_argument("gtest_args", nargs="*")
    args = parser.parse_args()

    names = list_tests(args.binary)

    def apply(path):
        """The tests the list at `path` leaves out, each with its reason, in list order."""
        found, stale = [], []
        for pattern, reason in read_list(path):
            hit = [n for n in names if matches(n, pattern)]
            if not hit:
                stale.append(pattern)
            found += [{"test": n, "reason": reason} for n in hit]
        if stale:
            fail(f"{path}: these patterns match no test (renamed or removed?): " + ", ".join(stale))
        return found

    # A test that two entries match is listed once, under the first.
    seen, unique = set(), []
    for item in apply(args.list):
        if item["test"] not in seen:
            seen.add(item["test"])
            unique.append(item)
    gaps = []
    for item in apply(args.gaps) if args.gaps else []:
        if item["test"] not in seen:
            seen.add(item["test"])
            gaps.append(item)

    if args.dry_run:
        by_reason = {}
        for item in unique:
            by_reason.setdefault(item["reason"], []).append(item["test"])
        print(f"{len(unique)} of {len(names)} tests do not apply:")
        for reason, tests in by_reason.items():
            print(f"  {len(tests):4}  {reason}")
        if args.gaps:
            print(f"{len(gaps)} more are known gaps:")
            for reason in dict.fromkeys(item["reason"] for item in gaps):
                print(f"  {sum(1 for i in gaps if i['reason'] == reason):4}  {reason}")
        return 0

    out_dir = os.path.dirname(os.path.abspath(args.xml))
    os.makedirs(out_dir, exist_ok=True)
    if os.environ.get("GTEST_SHARD_INDEX", "0") == "0":
        with open(os.path.join(out_dir, "not-applicable.json"), "w") as f:
            json.dump({"list": args.list, "tests": unique}, f, indent=1)
        if args.gaps:
            with open(os.path.join(out_dir, "known-gaps.json"), "w") as f:
                json.dump({"list": args.gaps, "tests": gaps}, f, indent=1)

    cmd = [args.binary, f"--gtest_output=xml:{args.xml}"]
    if seen:
        cmd.append("--gtest_filter=-" + ":".join(sorted(seen)))
    cmd += args.gtest_args
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
