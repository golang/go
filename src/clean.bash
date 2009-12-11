#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

GOBIN="${GOBIN:-$HOME/bin}"

rm -rf "$GOROOT"/pkg/${GOOS}_$GOARCH
rm -f "$GOROOT"/lib/*.a
for i in lib9 libbio libcgo libmach cmd pkg \
	../misc/cgo/gmp ../misc/cgo/stdio \
	../test/bench
do(
	cd "$GOROOT"/src/$i || exit 1
	if test -f clean.bash; then
		bash clean.bash
	else
		"$GOBIN"/gomake clean
	fi
)done
