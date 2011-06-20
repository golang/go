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
		time gomake install
	) || exit $i
fi

(xcd pkg
gomake testshort
) || exit $?

(xcd pkg/sync;
GOMAXPROCS=10 gomake testshort
) || exit $?

(xcd cmd/ebnflint
time gomake test
) || exit $?

(xcd cmd/godefs
gomake test
) || exit $?

[ "$GOARCH" == arm ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/stdio
gomake clean
./test.bash
) || exit $?

[ "$GOARCH" == arm ] ||
(xcd ../misc/cgo/life
gomake clean
./test.bash
) || exit $?

[ "$GOARCH" == arm ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/test
gomake clean
gotest
) || exit $?

(xcd pkg/exp/ogle
gomake clean
time gomake ogle
) || exit $?

(xcd ../doc/progs
time ./run
) || exit $?

[ "$GOARCH" == arm ] ||  # uses network, fails under QEMU
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
(xcd ../test/bench
./timing.sh -test
) || exit $?

[ "$GOHOSTOS" == windows ] ||
(xcd ../test
./run
) || exit $?

echo
echo ALL TESTS PASSED
