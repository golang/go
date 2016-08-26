#!/usr/bin/env bash
# Copyright 2016 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if [ ! -f src/host/host.go ]; then
	cwd=$(pwd)
	echo "misc/cgo/testplugin/test.bash is running in $cwd" 1>&2
	exit 1
fi

goos=$(go env GOOS)
goarch=$(go env GOARCH)

function cleanup() {
	rm -f plugin1.so host pkg
}
trap cleanup EXIT

rm -rf pkg

GOPATH=$(pwd) go build -buildmode=plugin plugin1
GOPATH=$(pwd) go build host

LD_LIBRARY_PATH=$(pwd) ./host
