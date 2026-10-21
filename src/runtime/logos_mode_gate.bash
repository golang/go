#!/usr/bin/env bash
# Copyright 2026 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Test-only runner for the -O tier checks of package runtime and
# runtime/pprof (optmode_logos_test.go in each).
#
# Usage: bash logos_mode_gate.bash plain|O1|O2|O3|all [go test flags]
#
# Each mode pairs a build configuration with the variable-make stack
# buffer, in bytes, that the compiler must give a package built with it
# (logos-docs T-105, T-302; cmd/go/internal/work/logos_opt.go sets
# variablemakethreshold per level, and the script test
# cmd/go/testdata/script/logos_opt_levels.txt pins the same pairs). The
# size is passed to the tests in LOGOS_TEST_EXPECT_STACKBUF. A test binary
# whose measured buffer differs from it fails, so a regressed allocation
# optimization cannot fall back to the upstream assertions.
#
#   plain  go test, no -O, variable unset: the upstream bodies run
#          unchanged, as for any ordinary go test of these packages.
#   O1     -O=1, 32 bytes: the upstream bodies run after the buffer is
#          checked to be the upstream size.
#   O2     -O=2, 256 bytes: the tier checks run.
#   O3     -O=3, 4096 bytes: the tier checks run.
#
# GO selects the command (default go). Plain go and logos accept the same
# -O flag; plain mode needs the plain go command because logos defaults
# to O2.

set -eu

gocmd=${GO:-go}
tests='^(TestAppendGrowthStack|TestAppendByteInLoop|TestAppendPtrInLoop|TestAppendByteCapInLoop|TestFunctionAlignmentTraceback|TestGenericsHashKeyInPprofBuilder)$'

usage() {
	echo "usage: $0 plain|O1|O2|O3|all [go test flags]" >&2
	exit 2
}

run() {
	mode=$1
	shift
	case $mode in
	plain)
		case $(basename "$gocmd") in
		logos | logos.exe)
			echo "$0: plain mode needs the go command: logos defaults to -O=2" >&2
			exit 2
			;;
		esac
		unset LOGOS_TEST_EXPECT_STACKBUF
		set -- -count=1 -run "$tests" "$@" runtime runtime/pprof
		;;
	O1) set -- -count=1 -O=1 -run "$tests" "$@" runtime runtime/pprof ;;
	O2) set -- -count=1 -O=2 -run "$tests" "$@" runtime runtime/pprof ;;
	O3) set -- -count=1 -O=3 -run "$tests" "$@" runtime runtime/pprof ;;
	*) usage ;;
	esac
	case $mode in
	O1) LOGOS_TEST_EXPECT_STACKBUF=32 ;;
	O2) LOGOS_TEST_EXPECT_STACKBUF=256 ;;
	O3) LOGOS_TEST_EXPECT_STACKBUF=4096 ;;
	esac
	export LOGOS_TEST_EXPECT_STACKBUF
	echo "== $gocmd test $mode (LOGOS_TEST_EXPECT_STACKBUF=${LOGOS_TEST_EXPECT_STACKBUF:-unset})"
	"$gocmd" test "$@"
}

[ $# -ge 1 ] || usage
mode=$1
shift
if [ "$mode" = all ]; then
	for m in plain O1 O2 O3; do
		run "$m" "$@"
	done
else
	run "$mode" "$@"
fi
