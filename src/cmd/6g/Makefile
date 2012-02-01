# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

TARG=6g

HFILES=\
	../gc/go.h\
	../6l/6.out.h\
	gg.h\
	opt.h\

OFILES=\
	../6l/enam.$O\
	cgen.$O\
	cplx.$O\
	galign.$O\
	ggen.$O\
	gobj.$O\
	gsubr.$O\
	list.$O\
	peep.$O\
	pgen.$O\
	reg.$O\

LIB=\
	../gc/gc.a$O\

include ../../Make.ctool

%.$O: ../gc/%.c
	$(HOST_CC) $(HOST_CFLAGS) -I. -o $@ ../gc/$*.c
