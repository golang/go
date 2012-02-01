# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

TARG=5g

HFILES=\
	../gc/go.h\
	../5l/5.out.h\
	gg.h\
	opt.h\

OFILES=\
	../5l/enam.$O\
	cgen.$O\
	cgen64.$O\
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
