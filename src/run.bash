#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

# no core files, please
ulimit -c 0

xcd() {
	echo
	echo --- cd $1
	builtin cd $1
}

maketest() {
	for i
	do
		(
			xcd $i
			gomake clean
			time gomake
			gomake install
			gomake test
		) || exit $?
	done
}

maketest \
	pkg \

# all of these are subtly different
# from what maketest does.

(xcd pkg/sync;
gomake clean;
time gomake
GOMAXPROCS=10 gomake test
) || exit $?

(xcd cmd/gofmt
gomake clean
time gomake
time gomake smoketest
) || exit $?

(xcd cmd/ebnflint
gomake clean
time gomake
time gomake test
) || exit $?

(xcd ../misc/cgo/stdio
gomake clean
./test.bash
) || exit $?

(xcd pkg/exp/ogle
gomake clean
time gomake ogle
) || exit $?

(xcd ../doc/progs
time ./run
) || exit $?

(xcd ../test/bench
./timing.sh -test
) || exit $?

(xcd ../test
./run
) || exit $?

