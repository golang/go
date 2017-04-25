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
	rm -f plugin*.so unnamed*.so iface*.so
	rm -rf host pkg sub iface issue18676 issue19534
}
trap cleanup EXIT

rm -rf pkg sub
mkdir sub

GOPATH=$(pwd) go build -buildmode=plugin plugin1
GOPATH=$(pwd) go build -buildmode=plugin plugin2
GOPATH=$(pwd)/altpath go build -buildmode=plugin plugin-mismatch
GOPATH=$(pwd) go build -buildmode=plugin -o=sub/plugin1.so sub/plugin1
GOPATH=$(pwd) go build -buildmode=plugin unnamed1.go
GOPATH=$(pwd) go build -buildmode=plugin unnamed2.go
GOPATH=$(pwd) go build host

LD_LIBRARY_PATH=$(pwd) ./host

# Test that types and itabs get properly uniqified.
GOPATH=$(pwd) go build -buildmode=plugin iface_a
GOPATH=$(pwd) go build -buildmode=plugin iface_b
GOPATH=$(pwd) go build iface
LD_LIBRARY_PATH=$(pwd) ./iface

# Test for issue 18676 - make sure we don't add the same itab twice.
# The buggy code hangs forever, so use a timeout to check for that.
GOPATH=$(pwd) go build -buildmode=plugin -o plugin.so src/issue18676/plugin.go
GOPATH=$(pwd) go build -o issue18676 src/issue18676/main.go
timeout 10s ./issue18676

# Test for issue 19534 - that we can load a plugin built in a path with non-alpha
# characters
GOPATH=$(pwd) go build -buildmode=plugin -ldflags='-pluginpath=issue.19534' -o plugin.so src/issue19534/plugin.go
GOPATH=$(pwd) go build -o issue19534 src/issue19534/main.go
./issue19534
