#!/usr/bin/env python3
"""Decides whether a pull request's or a push's changes need a workflow's test jobs.

Reads the changed paths (one per line) and the patterns of paths that cannot
affect a build or a test result. Prints run=false only when every changed path
matches those patterns and none matches the workflow's own files (--always).
Anything else, including an empty or oversized list, prints run=true: a skipped
job counts as passing for a required check, so an unclear case must test.

The decision goes to $GITHUB_OUTPUT and a short explanation to
$GITHUB_STEP_SUMMARY when those are set; both are printed otherwise.
"""

import argparse
import fnmatch
import os
import sys

# The pull request files API lists at most this many files; the compare API,
# used for a push, lists at most 300 (--file-limit).
API_FILE_LIMIT = 3000


def read_lines(path):
    """Returns the non-empty lines of a file, without comments."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines


def matches(path, pattern):
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def decide(paths, skippable, always, limit=API_FILE_LIMIT):
    """Returns (run, reason)."""
    if not paths:
        return True, "no changed files are listed"
    if len(paths) >= limit:
        return True, f"{len(paths)} changed files, at the API's limit, so the list may be incomplete"
    for path in paths:
        if any(matches(path, p) for p in always):
            return True, f"`{path}` is one of this workflow's own files"
    for path in paths:
        if not any(matches(path, p) for p in skippable):
            return True, f"`{path}` can affect a build or a test result"
    return False, f"every changed file ({len(paths)}) is in .github/no-test-paths.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("changed", help="file with the changed paths, one per line")
    parser.add_argument("skippable", help="file with the patterns that need no tests")
    parser.add_argument("--always", nargs="*", default=[],
                        help="patterns of this workflow's own files, which always need tests")
    parser.add_argument("--file-limit", type=int, default=API_FILE_LIMIT,
                        help="the most files the API that listed the changes returns")
    args = parser.parse_args()

    with open(args.changed) as f:
        paths = sorted({line.strip() for line in f if line.strip()})
    run, reason = decide(paths, read_lines(args.skippable), args.always, args.file_limit)

    decision = "true" if run else "false"
    verdict = "Tests run" if run else "Tests skipped"
    summary = f"**{verdict}:** {reason}.\n"
    print(f"run={decision}: {reason}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as f:
            f.write(f"run={decision}\n")
    report = os.environ.get("GITHUB_STEP_SUMMARY")
    if report:
        with open(report, "a") as f:
            f.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
