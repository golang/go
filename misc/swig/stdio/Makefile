# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=swig/file
SWIGFILES=\
	file.swig

CLEANFILES+=hello

include ../../../src/Make.pkg

%: install %.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) $*.go
	$(LD) $(SWIG_RPATH) -o $@ $*.$O
