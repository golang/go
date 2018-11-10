#!/bin/bash
# Copyright 2016 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# naclmake.bash builds runs make.bash for nacl, but not does run any
# tests. This is used by the continuous build.

# Assumes that sel_ldr binaries and go_nacl_$GOARCH_exec scripts are in $PATH;
# see ../misc/nacl/README.

set -e
ulimit -c 0

# guess GOARCH if not set
naclGOARCH=$GOARCH
if [ -z "$naclGOARCH" ]; then
	case "$(uname -m)" in
	x86_64)
		naclGOARCH=amd64p32
		;;
	armv7l) # NativeClient on ARM only supports ARMv7A.
		naclGOARCH=arm
		;;
	i?86)
		naclGOARCH=386
		;;
	esac
fi

unset GOOS GOARCH
if [ ! -f make.bash ]; then
	echo 'nacltest.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

# the builder might have set GOROOT_FINAL.
export GOROOT=$(pwd)/..

# Build zip file embedded in package syscall.
echo "##### Building fake file system zip for nacl"
rm -f syscall/fstest_nacl.go
GOROOT_BOOTSTRAP=${GOROOT_BOOTSTRAP:-$HOME/go1.4}
gobin=$GOROOT_BOOTSTRAP/bin
GOROOT=$GOROOT_BOOTSTRAP $gobin/go run ../misc/nacl/mkzip.go -p syscall -r .. ../misc/nacl/testzip.proto syscall/fstest_nacl.go

# Run standard build and tests.
GOOS=nacl GOARCH=$naclGOARCH ./make.bash
