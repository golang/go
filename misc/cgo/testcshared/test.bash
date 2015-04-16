#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

function cleanup() {
	rm libgo.so testp
}
trap cleanup EXIT

GOPATH=$(pwd) go build -buildmode=c-shared -o libgo.so src/libgo/libgo.go

$(go env CC) $(go env GOGCCFLAGS) -o testp main0.c libgo.so
output=$(LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. ./testp)
# testp prints PASS at the end of its execution.
if [ "$output" != "PASS" ]; then
	echo "FAIL: got $output"
	exit 1
fi

$(go env CC) $(go env GOGCCFLAGS) -o testp main1.c -ldl
output=$(./testp ./libgo.so) 
# testp prints PASS at the end of its execution.
if [ "$output" != "PASS" ]; then
	echo "FAIL: got $output"
	exit 1
fi
