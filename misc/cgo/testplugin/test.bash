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
	rm -f plugin*.so unnamed*.so iface*.so issue*
	rm -rf host pkg sub iface
}
trap cleanup EXIT

rm -rf pkg sub
mkdir sub

GOPATH=$(pwd) go build -i -gcflags "$GO_GCFLAGS" -buildmode=plugin plugin1
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin plugin2
cp plugin2.so plugin2-dup.so
GOPATH=$(pwd)/altpath go build -gcflags "$GO_GCFLAGS" -buildmode=plugin plugin-mismatch
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o=sub/plugin1.so sub/plugin1
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o=unnamed1.so unnamed1/main.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o=unnamed2.so unnamed2/main.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" host

LD_LIBRARY_PATH=$(pwd) ./host

# Test that types and itabs get properly uniqified.
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin iface_a
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin iface_b
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" iface
LD_LIBRARY_PATH=$(pwd) ./iface

function _timeout() (
	set -e
	$2 &
	p=$!
	(sleep $1; kill $p 2>/dev/null) &
	p2=$!
	wait $p 2>/dev/null
	kill -0 $p2 2>/dev/null
)

# Test for issue 18676 - make sure we don't add the same itab twice.
# The buggy code hangs forever, so use a timeout to check for that.
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o plugin.so src/issue18676/plugin.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue18676 src/issue18676/main.go
_timeout 10s ./issue18676

# Test for issue 19534 - that we can load a plugin built in a path with non-alpha
# characters
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -ldflags='-pluginpath=issue.19534' -o plugin.so src/issue19534/plugin.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue19534 src/issue19534/main.go
./issue19534

# Test for issue 18584
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o plugin.so src/issue18584/plugin.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue18584 src/issue18584/main.go
./issue18584

# Test for issue 19418
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin "-ldflags=-X main.Val=linkstr" -o plugin.so src/issue19418/plugin.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue19418 src/issue19418/main.go
./issue19418

# Test for issue 19529
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o plugin.so src/issue19529/plugin.go

# Test for issue 22175
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o issue22175_plugin1.so src/issue22175/plugin1.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o issue22175_plugin2.so src/issue22175/plugin2.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue22175 src/issue22175/main.go
./issue22175

# Test for issue 22295
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o issue.22295.so issue22295.pkg
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue22295 src/issue22295.pkg/main.go
./issue22295

# Test for issue 24351
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -buildmode=plugin -o issue24351.so src/issue24351/plugin.go
GOPATH=$(pwd) go build -gcflags "$GO_GCFLAGS" -o issue24351 src/issue24351/main.go
./issue24351
