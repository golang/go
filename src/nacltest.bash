#!/bin/bash
# Copyright 2014 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing Native Client on builders or locally.
# Builds a test file system and embeds it into package syscall
# in every generated binary.
#
# Assumes that sel_ldr binaries are in $PATH; see ../misc/nacl/README.

set -e
ulimit -c 0

# Check GOARCH.
naclGOARCH=${GOARCH:-386}
case "$naclGOARCH" in
amd64p32)
	if ! which sel_ldr_x86_64 >/dev/null; then
		echo 'cannot find sel_ldr_x86_64' 1>&2
		exit 1
	fi
	;;
386)
	if ! which sel_ldr_x86_32 >/dev/null; then
		echo 'cannot find sel_ldr_x86_32' 1>&2
		exit 1
	fi
	;;
*)
	echo 'unsupported $GOARCH for nacl: '"$naclGOARCH" 1>&2
	exit 1
esac

# Run host build to get toolchain for running zip generator.
unset GOOS GOARCH
if [ ! -f make.bash ]; then
	echo 'nacl.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH ./make.bash

# Build zip file embedded in package syscall.
gobin=${GOBIN:-$(pwd)/../bin}
rm -f pkg/syscall/fstest_nacl.go
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH $gobin/go run ../misc/nacl/mkzip.go -p syscall -r .. ../misc/nacl/testzip.proto pkg/syscall/fstest_nacl.go

# Run standard build and tests.
export PATH=$(pwd)/../misc/nacl:$PATH
GOOS=nacl GOARCH=$naclGOARCH ./all.bash --no-clean
