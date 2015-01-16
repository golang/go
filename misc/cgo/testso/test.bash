#!/usr/bin/env bash
# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if [ "$(uname -m)" == ppc64 -o "$(uname -m)" == ppc64le ]; then
	# External linking not implemented on ppc64
	echo "skipping test on ppc64 (issue #8912)"
	exit
fi

args=
dyld_envvar=LD_LIBRARY_PATH
ext=so
if [ "$(uname)" == "Darwin" ]; then
	args="-undefined suppress -flat_namespace"
	dyld_envvar=DYLD_LIBRARY_PATH
	ext=dylib
fi

dylib=libcgosotest.$ext
$(go env CC) $(go env GOGCCFLAGS) -shared $args -o $dylib cgoso_c.c
go build main.go

eval "$dyld_envvar"=. ./main
rm -rf $dylib main *.dSYM
