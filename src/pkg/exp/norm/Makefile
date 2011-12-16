# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=exp/norm
GOFILES=\
	composition.go\
	input.go\
	forminfo.go\
	normalize.go\
	readwriter.go\
	tables.go\
	trie.go\

include ../../../Make.pkg

CLEANFILES+=maketables maketesttables

maketables: maketables.go triegen.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) maketables.go triegen.go
	$(LD) -o maketables maketables.$O

maketesttables: maketesttables.go triegen.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) maketesttables.go triegen.go
	$(LD) -o maketesttables maketesttables.$O

normregtest: normregtest.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) normregtest.go
	$(LD) -o normregtest normregtest.$O

tables:	maketables
	./maketables > tables.go
	gofmt -w tables.go

trietesttables: maketesttables
	./maketesttables > triedata_test.go
	gofmt -w triedata_test.go

# Build (but do not run) maketables during testing,
# just to make sure it still compiles.
testshort: maketables maketesttables

# Downloads from www.unicode.org, so not part
# of standard test scripts.
test: testtables regtest

testtables: maketables
	./maketables -test -tables=

regtest: normregtest
	./normregtest
