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
goarch=$(go env GOARCH)

# Directory where cgo headers and outputs will be installed.
# The installation directory format varies depending on the platform.
installdir=pkg/${goos}_${goarch}_testcshared_shared
if [ "${goos}/${goarch}" == "android/arm" ] || [ "${goos}/${goarch}" == "darwin/amd64" ]; then
	installdir=pkg/${goos}_${goarch}_testcshared
fi

# Temporary directory on the android device.
androidpath=/data/local/tmp/testcshared-$$

function cleanup() {
	rm -rf libgo.$libext libgo2.$libext libgo.h testp testp2 testp3 pkg

	rm -rf $(go env GOROOT)/${installdir}

	if [ "$goos" == "android" ]; then
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
		output=$(adb shell "cd ${androidpath}; $@")
		output=$(echo $output|tr -d '\r')
		case $output in
			*PASS) echo "PASS";; 
			*) echo "$output";;
		esac
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

rm -rf pkg

suffix="-installsuffix testcshared"

libext="so"
if [ "$goos" == "darwin" ]; then
	libext="dylib"
fi

# Create the header files.
GOPATH=$(pwd) go install -buildmode=c-shared $suffix libgo

GOPATH=$(pwd) go build -buildmode=c-shared $suffix -o libgo.$libext src/libgo/libgo.go
binpush libgo.$libext

# test0: exported symbols in shared lib are accessible.
# TODO(iant): using _shared here shouldn't really be necessary.
$(go env CC) $(go env GOGCCFLAGS) -I ${installdir} -o testp main0.c libgo.$libext
binpush testp

output=$(run LD_LIBRARY_PATH=. ./testp)
if [ "$output" != "PASS" ]; then
	echo "FAIL test0 got ${output}"
	exit 1
fi

# test1: shared library can be dynamically loaded and exported symbols are accessible.
$(go env CC) $(go env GOGCCFLAGS) -o testp main1.c -ldl
binpush testp
output=$(run ./testp ./libgo.$libext)
if [ "$output" != "PASS" ]; then
	echo "FAIL test1 got ${output}"
	exit 1
fi

# test2: tests libgo2 which does not export any functions.
GOPATH=$(pwd) go build -buildmode=c-shared $suffix -o libgo2.$libext src/libgo2/libgo2.go
binpush libgo2.$libext
linkflags="-Wl,--no-as-needed"
if [ "$goos" == "darwin" ]; then
	linkflags=""
fi
$(go env CC) $(go env GOGCCFLAGS) -o testp2 main2.c $linkflags libgo2.$libext
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
