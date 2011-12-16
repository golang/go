# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=unicode
GOFILES=\
	casetables.go\
	digit.go\
	graphic.go\
	letter.go\
	tables.go\

include ../../Make.pkg

CLEANFILES+=maketables

maketables: maketables.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) maketables.go
	$(LD) -o maketables maketables.$O

tables:	maketables
	./maketables --tables=all > tables.go
	gofmt -w tables.go

# Build (but do not run) maketables during testing,
# just to make sure it still compiles.
test: maketables
testshort: maketables

# Downloads from www.unicode.org, so not part
# of standard test scripts.
testtables: maketables
	@echo '***' Be sure to make tables and make install first
	./maketables -test
