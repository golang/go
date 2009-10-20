#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

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
			make clean
			time make
			make install
			make test
		) || exit $?
	done
}

maketest \
	pkg \

# all of these are subtly different
# from what maketest does.

(xcd pkg/sync;
make clean;
time make
GOMAXPROCS=10 make test
) || exit $?

(xcd cmd/gofmt
make clean
time make
time make smoketest
) || exit $?

(xcd cmd/ebnflint
make clean
time make
time make test
) || exit $?

(xcd ../misc/cgo/stdio
make clean
./test.bash
) || exit $?

(xcd ../usr/r/rpc
make clean
time make
./chanrun
) || exit $?

(xcd pkg/exp/ogle
make clean
time make ogle
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

