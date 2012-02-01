# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

LIB=cc.a$O

HFILES=\
	cc.h\
	y.tab.h\

YFILES=\
	cc.y\

OFILES=\
	y.tab.$O\
	lex.$O\
	mac.$O\
	dcl.$O\
	acid.$O\
	godefs.$O\
	bits.$O\
	com.$O\
	scon.$O\
	funct.$O\
	sub.$O\
	com64.$O\
	dpchk.$O\
	omachcap.$O\

NOINSTALL=1
include ../../Make.clib

install: $(LIB)
