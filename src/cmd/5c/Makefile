# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

TARG=5c

HFILES=\
	gc.h\
	../5l/5.out.h\
	../cc/cc.h\

OFILES=\
	cgen.$O\
	list.$O\
	sgen.$O\
	swt.$O\
	txt.$O\
	mul.$O\
	reg.$O\
	peep.$O\
	pgen.$O\
	pswt.$O\
	../5l/enam.$O\

LIB=\
	../cc/cc.a\

include ../../Make.ctool

%.$O: ../cc/%.c
	$(HOST_CC) $(HOST_CFLAGS) -I. -o $@ ../cc/$*.c
