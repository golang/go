# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CLEANFILES+=maketables

maketables: maketables.go
	go build $^

tables:	maketables
	./maketables > tables.go
	gofmt -w -s tables.go

# Build (but do not run) maketables during testing,
# just to make sure it still compiles.
testshort: maketables
