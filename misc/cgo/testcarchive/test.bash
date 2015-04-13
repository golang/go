#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

ccargs=""
if [ "$(go env GOOS)" == "darwin" ]; then
	ccargs="-Wl,-no_pie"
	# For darwin/arm.
	# TODO(crawshaw): Can we do better?
	ccargs="$ccargs -framework CoreFoundation"
fi

# TODO(crawshaw): Consider a go env for exec script name.
bin=./testp
exec_script=go_$(go env GOOS)_$(go env GOARCH)_exec
if [ "$(which $exec_script)" != "" ]; then
	bin="$exec_script ./testp"
fi

GOPATH=$(pwd) go build -buildmode=c-archive src/libgo/libgo.go
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
$bin
rm libgo.a testp

GOPATH=$(pwd) go build -buildmode=c-archive -o libgo.a libgo
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
$bin
rm libgo.a testp
