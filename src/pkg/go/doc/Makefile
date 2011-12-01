# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=go/doc
GOFILES=\
	comment.go\
	doc.go\
	example.go\

include ../../../Make.pkg

# Script to test heading detection heuristic
CLEANFILES+=headscan
headscan: headscan.go
	$(GC) headscan.go
	$(LD) -o headscan headscan.$(O)
