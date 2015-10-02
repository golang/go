#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

ccargs=
if [ "$(go env GOOS)" == "darwin" ]; then
	ccargs="-Wl,-no_pie"
	# For darwin/arm.
	# TODO(crawshaw): Can we do better?
	ccargs="$ccargs -framework CoreFoundation -framework Foundation"
fi
ccargs="$ccargs -I pkg/$(go env GOOS)_$(go env GOARCH)"

# TODO(crawshaw): Consider a go env for exec script name.
bin=./testp
exec_script=go_$(go env GOOS)_$(go env GOARCH)_exec
if [ "$(which $exec_script)" != "" ]; then
	bin="$exec_script ./testp"
fi

rm -rf libgo.a libgo.h testp pkg

# Installing first will create the header files we want.

GOPATH=$(pwd) go install -buildmode=c-archive libgo
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c pkg/$(go env GOOS)_$(go env GOARCH)/libgo.a
$bin arg1 arg2
rm -f libgo.a libgo.h testp

# Test building libgo other than installing it.
# Header files are now present.

GOPATH=$(pwd) go build -buildmode=c-archive src/libgo/libgo.go
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
$bin arg1 arg2
rm -f libgo.a libgo.h testp

GOPATH=$(pwd) go build -buildmode=c-archive -o libgo.a libgo
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
$bin arg1 arg2
rm -rf libgo.a libgo.h testp pkg
