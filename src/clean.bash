#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if [ -z "$GOROOT" ] ; then
	echo '$GOROOT not set'
	exit 1
fi
if [ -z "$GOOS" ] ; then
	echo '$GOOS not set'
	exit 1
fi
if [ -z "$GOARCH" ] ; then
	echo '$GOARCH not set'
	exit 1
fi

GOBIN="${GOBIN:-$HOME/bin}"

if [ "$1" != "--nopkg" ]; then
	rm -rf "$GOROOT"/pkg/${GOOS}_$GOARCH
fi
rm -f "$GOROOT"/lib/*.a
for i in lib9 libbio libcgo libmach cmd pkg \
	../misc/cgo/gmp ../misc/cgo/stdio \
	../test/bench ../test/garbage
do(
	cd "$GOROOT"/src/$i || exit 1
	if test -f clean.bash; then
		bash clean.bash
	else
		"$GOBIN"/gomake clean
	fi
)done
