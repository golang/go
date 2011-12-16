# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=goyacc
GOFILES=\
	goyacc.go\

include ../../Make.cmd

units: goyacc units.y
	./goyacc -p units_ units.y
	$(GC) $(GCFLAGS) $(GCIMPORTS) y.go
	$(LD) -o units y.$O

CLEANFILES += units y.go y.output
