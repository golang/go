#!/bin/bash
# Copyright 2014 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing Native Client on builders or locally.
# Builds a test file system and embeds it into package syscall
# in every generated binary.
#
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

# Check GOARCH.
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
arm)
	if ! which sel_ldr_arm >/dev/null; then
		echo 'cannot find sel_ldr_arm' 1>&2
		exit 1
	fi
	;;
*)
	echo 'unsupported $GOARCH for nacl: '"$naclGOARCH" 1>&2
	exit 1
esac

if ! which go_nacl_${naclGOARCH}_exec >/dev/null; then
	echo "cannot find go_nacl_${naclGOARCH}_exec, see ../misc/nacl/README." 1>&2
	exit 1
fi

# Run host build to get toolchain for running zip generator.
unset GOOS GOARCH
if [ ! -f make.bash ]; then
	echo 'nacl.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH ./make.bash

# the builder might have set GOROOT_FINAL.
export GOROOT=$(pwd)/..

# Build zip file embedded in package syscall.
gobin=${GOBIN:-$(pwd)/../bin}
rm -f syscall/fstest_nacl.go
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH $gobin/go run ../misc/nacl/mkzip.go -p syscall -r .. ../misc/nacl/testzip.proto syscall/fstest_nacl.go

# Run standard build and tests.
export PATH=$(pwd)/../misc/nacl:$PATH
GOOS=nacl GOARCH=$naclGOARCH ./all.bash --no-clean
