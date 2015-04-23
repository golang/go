#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing Android, this script requires adb to push and run compiled
# binaries on a target device.

set -e

if [ ! -f src/libgo/libgo.go ]; then
	cwd=$(pwd)
	echo 'misc/cgo/testcshared/test.bash is running in $cwd' 1>&2
	exit 1
fi

goos=$(go env GOOS)

# Temporary directory on the android device.
androidpath=/data/local/tmp/testcshared-$$

function cleanup() {
	rm -f libgo.so libgo2.so testp testp2 testp3

	if [ "$(go env GOOS)" == "android" ]; then
		adb shell rm -rf $androidpath
	fi
}
trap cleanup EXIT

if [ "$goos" == "android" ]; then
	adb shell mkdir -p "$androidpath"
fi

function run() {
	case "$goos" in
	"android")
		local args=$@
		for ((i=0; i < ${#args}; i++)); do
			args[$i]=${args[$i]//.\//${androidpath}\/}
			args[$i]=${args[$i]//=./=${androidpath}}
		done
		echo $(adb shell ${args} | tr -d '\r')
		;;
	*)
		echo $(env $@)
		;;
	esac
}

function binpush() {
	bin=${1}
	if [ "$goos" == "android" ]; then
		adb push "$bin"  "${androidpath}/${bin}" 2>/dev/null
	fi
}

GOPATH=$(pwd) go build -buildmode=c-shared -o libgo.so src/libgo/libgo.go
binpush libgo.so

# test0: exported symbols in shared lib are accessible.
$(go env CC) $(go env GOGCCFLAGS) -o testp main0.c libgo.so
binpush testp
output=$(run LD_LIBRARY_PATH=. ./testp)
if [ "$output" != "PASS" ]; then
	echo "FAIL test0 got ${output}"
	exit 1
fi

# test1: .so can be dynamically loaded and exported symbols are accessible.
$(go env CC) $(go env GOGCCFLAGS) -o testp main1.c -ldl
binpush testp
output=$(run ./testp ./libgo.so)
if [ "$output" != "PASS" ]; then
	echo "FAIL test1 got ${output}"
	exit 1
fi

# test2: tests libgo2.so which does not export any functions.
GOPATH=$(pwd) go build -buildmode=c-shared -o libgo2.so src/libgo2/libgo2.go
binpush libgo2.so
$(go env CC) $(go env GOGCCFLAGS) -o testp2 main2.c -Wl,--no-as-needed libgo2.so
binpush testp2
output=$(run LD_LIBRARY_PATH=. ./testp2)
if [ "$output" != "PASS" ]; then
	echo "FAIL test2 got ${output}"
	exit 1
fi

# test3: tests main.main is exported on android.
if [ "$goos" == "android" ]; then
	$(go env CC) $(go env GOGCCFLAGS) -o testp3 main3.c -ldl
	binpush testp3
	output=$(run ./testp ./libgo.so)
	if [ "$output" != "PASS" ]; then
		echo "FAIL test3 got ${output}"
		exit 1
	fi
fi
echo "ok"
