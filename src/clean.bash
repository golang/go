#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if [ ! -f env.bash ]; then
	echo 'clean.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
. ./env.bash
if [ ! -f Make.inc ] ; then
    GOROOT_FINAL=${GOROOT_FINAL:-$GOROOT}
    sed 's!@@GOROOT@@!'"$GOROOT_FINAL"'!' Make.inc.in >Make.inc
fi

if [ "$1" != "--nopkg" ]; then
	rm -rf "$GOROOT"/pkg/${GOOS}_$GOARCH
fi
rm -f "$GOROOT"/lib/*.a
for i in lib9 libbio libmach cmd pkg \
	../misc/cgo/gmp ../misc/cgo/stdio \
	../misc/cgo/life ../misc/cgo/test \
	../misc/dashboard/builder ../misc/goplay\
	../doc/codelab/wiki\
	../test/bench/shootout ../test/garbage
do
	# Do not use gomake here. It may not be available.
	$MAKE -C "$GOROOT/src/$i" clean
done
