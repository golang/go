#!/bin/sh
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

go install # So the next line will produce updated documentation.
go help documentation | sed 's; \*/; * /;' >doc.go
gofmt -w doc.go

