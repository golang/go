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

maketest() {
	for i
	do
		(
			xcd $i
			if $rebuild; then
				gomake clean
				time gomake
				gomake install
			fi
			gomake test
		) || exit $?
	done
}

maketest \
	pkg \

# all of these are subtly different
# from what maketest does.

(xcd pkg/sync;
if $rebuild; then
	gomake clean;
	time gomake
fi
GOMAXPROCS=10 gomake test
) || exit $?

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

