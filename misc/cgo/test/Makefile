# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../src/Make.inc

TARG=runtime/cgotest

CGOFILES=\
	align.go\
	basic.go\
	callback.go\
	env.go\
	exports.go\
	helpers.go\
	issue1222.go\
	issue1328.go\
	issue1560.go\
	issue2462.go\
	duplicate_symbol.go\

CGO_OFILES=\
	callback_c.o\

OFILES=\
	runtime.$O\

include ../../../src/Make.pkg
