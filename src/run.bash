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
			make test
		) || exit $?
	done
}

maketest \
	lib/fmt\
	lib/hash\
	lib/json\
	lib/math\
	lib/net\
	lib/reflect\
	lib/regexp\
	lib/strconv\
	lib/tabwriter\
	lib/time\

# all of these are subtly different
# from what maketest does.

(xcd lib; make test) || exit $?

(xcd lib/sync;
make clean;
time make
GOMAXPROCS=10 make test
) || exit $?

(xcd ../usr/gri/pretty
make clean
time make
make smoketest
) || exit $?

(xcd ../usr/gri/gosrc
make clean
time make
# make test
) || exit $?

(xcd ../doc/progs
time run
) || exit $?

(xcd ../test
./run
) || exit $?

