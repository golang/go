#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

rm -rf $GOROOT/pkg/${GOOS}_$GOARCH
rm -f $GOROOT/lib/*.a
for i in lib9 libbio libcgo libmach libregexp cmd pkg \
	../misc/cgo/gmp ../misc/cgo/stdio \
	../usr/r/rpc ../usr/dsymonds/iterable \
	../usr/austin/eval ../usr/austin/ogle ../test/bench
do(
	cd $i
	if test -f clean.bash; then
		bash clean.bash
	else
		make clean
	fi
)done
