#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

GOBIN="${GOBIN:-$HOME/bin}"

# no core files, please
ulimit -c 0

xcd() {
	echo
	echo --- cd $1
	builtin cd "$GOROOT"/src/$1
}

maketest() {
	for i
	do
		(
			xcd $i
			"$GOBIN"/gomake clean
			time "$GOBIN"/gomake
			"$GOBIN"/gomake install
			"$GOBIN"/gomake test
		) || exit $?
	done
}

maketest \
	pkg \

# all of these are subtly different
# from what maketest does.

(xcd pkg/sync;
"$GOBIN"/gomake clean;
time "$GOBIN"/gomake
GOMAXPROCS=10 "$GOBIN"/gomake test
) || exit $?

(xcd cmd/gofmt
"$GOBIN"/gomake clean
time "$GOBIN"/gomake
time "$GOBIN"/gomake smoketest
) || exit $?

(xcd cmd/ebnflint
"$GOBIN"/gomake clean
time "$GOBIN"/gomake
time "$GOBIN"/gomake test
) || exit $?

(xcd ../misc/cgo/stdio
"$GOBIN"/gomake clean
./test.bash
) || exit $?

(xcd pkg/exp/ogle
"$GOBIN"/gomake clean
time "$GOBIN"/gomake ogle
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

