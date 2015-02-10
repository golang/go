#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

go build -o go.latest
./go.latest help documentation | sed 's; \*/; * /;' >doc.go
gofmt -w doc.go
rm go.latest

