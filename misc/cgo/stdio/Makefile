# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=stdio
CGOFILES=\
	file.go\

CLEANFILES+=hello fib chain run.out

include ../../../src/Make.pkg

%: install %.go
	$(GC) $(GCFLAGS) $(GCIMPORTS) $*.go
	$(LD) -o $@ $*.$O
