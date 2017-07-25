#!/bin/bash

# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e -o -x

LDFLAGS="-X main.version=$(git describe --always --dirty='*')"

GOOS=windows GOARCH=386 go build -o build/installer.exe    -ldflags="$LDFLAGS"
GOOS=linux GOARCH=386   go build -o build/installer_linux  -ldflags="$LDFLAGS"
GOOS=darwin GOARCH=386  go build -o build/installer_darwin -ldflags="$LDFLAGS"
