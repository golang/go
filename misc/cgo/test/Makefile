# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=runtime/cgotest

CGOFILES=\
	align.go\
	basic.go\
	callback.go\
	issue1222.go\
	issue1328.go\
	issue1560.go\

CGO_OFILES=\
	callback_c.o\

OFILES=\
	runtime.$O\

include ../../../src/Make.pkg
