#!/bin/bash
# Copyright 2014 The Go Authors. All rights reserved.
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

. ./naclmake.bash

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

export PATH=$(pwd)/../bin:$(pwd)/../misc/nacl:$PATH
GOROOT=$(../bin/go env GOROOT)
GOOS=nacl GOARCH=$naclGOARCH go tool dist test --no-rebuild

rm -f syscall/fstest_nacl.go
