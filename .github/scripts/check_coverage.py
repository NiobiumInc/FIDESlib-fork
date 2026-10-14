#!/usr/bin/env python3
"""Checks that every test runs in at least one configuration.

    check_coverage.py --binary BINARY [--sources FILE...] [--lists DIR [--gaps DIR]]

Tests are registered per backend (test/TestRegistration.hpp), so a test can end up registered on no
backend at all, and a not-applicable list can leave out the last backend that ran one. Two checks:

1. --sources: the test files the binary is built from. Every test a source file declares with
   FIDES_TEST_IF or FIDES_TEST_F_IF, and every TEST_P suite declared to have no cases somewhere
   (GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST), must be defined by the binary in at least one of
   the configurations a CI run uses: the CPU, haze and CUDA backends, and the CUDA backend in the
   concurrent-operations mode. A condition that no configuration satisfies (a typing mistake, a
   backend that no longer exists) makes the test run nowhere, and is an error. Listing the tests
   starts no backend, so this needs no GPU.
2. --lists: DIR holds cpu.txt, haze.txt and cuda.txt, the not-applicable lists of the three backends
   (run_gtest.py describes the format), and --gaps the directory of the known-gaps lists (default:
   tests-known-gaps next to DIR). A test that every one of those lists leaves out runs nowhere: an
   entry added to one list took away the last place it ran. This looks at the tests of the binary,
   so it holds for a binary the CPU and haze backends run (not for the CUDA test binary, which
   neither of them runs).

Exit status: 0 when every test runs somewhere, 1 when one does not, 2 for a malformed list.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gtest  # noqa: E402

BACKENDS = ("cpu", "haze", "cuda")

# The environments a CI run lists the tests under. The condition of a test reads nothing else.
CONFIGS = {
    "cpu": {"FIDESLIB_TEST_BACKEND": "cpu"},
    "haze": {"FIDESLIB_TEST_BACKEND": "haze"},
    "cuda": {"FIDESLIB_TEST_BACKEND": "cuda", "FIDESLIB_CONCURRENT_OPS": ""},
    "cuda, concurrent operations": {"FIDESLIB_TEST_BACKEND": "cuda", "FIDESLIB_CONCURRENT_OPS": "1"},
}

CONDITIONAL_TEST = re.compile(r"FIDES_TEST_(?:F_)?IF\(\s*(?:[^(),]|\([^()]*\))+,\s*(\w+)\s*,\s*(\w+)\s*\)")
EMPTY_SUITE = re.compile(r"GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST\(\s*(\w+)\s*\)")


def declared(sources):
    """(tests, suites) the source files declare as registered only under a condition."""
    tests, suites = {}, {}
    for path in sources:
        with open(path, errors="replace") as f:
            text = f.read()
        for suite, name in CONDITIONAL_TEST.findall(text):
            tests[f"{suite}.{name}"] = path
        for suite in EMPTY_SUITE.findall(text):
            suites[suite] = path
    return tests, suites


def check_registration(binary, sources):
    tests, suites = declared(sources)
    defined = set()
    for env in CONFIGS.values():
        defined.update(run_gtest.list_tests(binary, env))
    defined_suites = {name.split(".", 1)[0].rsplit("/", 1)[-1] for name in defined}
    missing = [f"{name} ({path}) is registered on no backend, so it runs nowhere"
               for name, path in sorted(tests.items()) if name not in defined]
    missing += [f"{suite} ({path}) has no case on any backend, so it runs nowhere"
                for suite, path in sorted(suites.items()) if suite not in defined_suites]
    print(f"{len(tests)} tests and {len(suites)} suites registered under a condition; {len(missing)} registered nowhere")
    for item in missing:
        print(f"::error::{item}")
    return len(missing)


def check_lists(binary, lists, gaps_dir):
    gaps_dir = gaps_dir or os.path.join(os.path.dirname(os.path.abspath(lists)), "tests-known-gaps")
    entries = {}
    for b in BACKENDS:
        entries[b] = run_gtest.read_list(os.path.join(lists, f"{b}.txt"))
        gaps = os.path.join(gaps_dir, f"{b}.txt")
        if os.path.exists(gaps):
            entries[b] += run_gtest.read_list(gaps)
    names = run_gtest.list_tests(binary, CONFIGS["cpu"])

    def leaves_out(backend, name):
        return any(run_gtest.matches(name, pattern) for pattern, _ in entries[backend])

    nowhere = [n for n in names if all(leaves_out(b, n) for b in BACKENDS)]
    print(f"{len(names)} tests in {os.path.basename(binary)}; {len(nowhere)} left out by every list")
    for name in nowhere:
        print(f"::error::{name} is left out by the list of every backend (not-applicable or known-gaps), so it runs nowhere")
    return len(nowhere)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--sources", nargs="*", default=[], help="test source files (globs allowed) the binary is built from")
    parser.add_argument("--lists", help="directory with cpu.txt, haze.txt and cuda.txt")
    parser.add_argument("--gaps", help="directory with the known-gaps lists")
    args = parser.parse_args()
    if not args.sources and not args.lists:
        parser.error("nothing to check: give --sources, --lists or both")

    failures = 0
    if args.sources:
        files = sorted({f for pattern in args.sources for f in glob.glob(pattern)})
        if not files:
            sys.exit("--sources matched no file")
        failures += check_registration(args.binary, files)
    if args.lists:
        failures += check_lists(args.binary, args.lists, args.gaps)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
