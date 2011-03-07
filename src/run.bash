#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
if [ "$1" = "--no-env" ]; then
	# caller has already run env.bash
	shift
else
	. ./env.bash
fi

unset MAKEFLAGS  # single-threaded make
unset CDPATH	# in case user has it set

# no core files, please
ulimit -c 0

# allow make.bash to avoid double-build of everything
rebuild=true
if [ "$1" = "--no-rebuild" ]; then
	rebuild=false
	shift
fi
		
xcd() {
	echo
	echo --- cd $1
	builtin cd "$GOROOT"/src/$1
}

if $rebuild; then
	(xcd pkg
		gomake clean
		time gomake
		gomake install
	) || exit $i
fi

(xcd pkg
gomake test
) || exit $?

(xcd pkg/sync;
if $rebuild; then
	gomake clean;
	time gomake
fi
GOMAXPROCS=10 gomake test
) || exit $?

[ "$GOARCH" == arm ] ||
(xcd cmd/gofmt
if $rebuild; then
	gomake clean;
	time gomake
fi
time gomake smoketest
) || exit $?

(xcd cmd/ebnflint
if $rebuild; then
	gomake clean;
	time gomake
fi
time gomake test
) || exit $?

[ "$GOARCH" == arm ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/stdio
gomake clean
./test.bash
) || exit $?

[ "$GOARCH" == arm ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/life
gomake clean
./test.bash
) || exit $?

(xcd pkg/exp/ogle
gomake clean
time gomake ogle
) || exit $?

[ "$GOHOSTOS" == windows ] ||
(xcd ../doc/progs
time ./run
) || exit $?

(xcd ../doc/codelab/wiki
gomake clean
gomake
gomake test
) || exit $?

for i in ../misc/dashboard/builder ../misc/goplay
do
	(xcd $i
	gomake clean
	gomake
	) || exit $?
done

[ "$GOARCH" == arm ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../test/bench
./timing.sh -test
) || exit $?

[ "$GOHOSTOS" == windows ] ||
(xcd ../test
./run
) || exit $?

echo
echo ALL TESTS PASSED
