# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=vet
GOFILES=\
	main.go\
	method.go\
	print.go\
	structtag.go\

include ../../Make.tool

test testshort: $(TARG)
	../../../test/errchk $(TARG) -printfuncs='Warn:1,Warnf:1' print.go
