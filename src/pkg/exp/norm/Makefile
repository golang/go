# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=exp/norm
GOFILES=\
	tables.go\

include ../../../Make.pkg

CLEANFILES+=maketables

maketables: maketables.go
	$(GC) maketables.go
	$(LD) -o maketables maketables.$O

tables:	maketables
	./maketables > tables.go
	gofmt -w tables.go

# Build (but do not run) maketables during testing,
# just to make sure it still compiles.
testshort: maketables

# Downloads from www.unicode.org, so not part
# of standard test scripts.
test: testtables

testtables: maketables
	./maketables -test -tables=
