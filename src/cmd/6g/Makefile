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
	list.$O\
	gobj.$O\
	galign.$O\
	ggen.$O\
	cgen.$O\
	cplx.$O\
	gsubr.$O\
	peep.$O\
	reg.$O\
	../6l/enam.$O\

LIB=\
	../gc/gc.a\

include ../../Make.ccmd

%.$O: ../gc/%.c
	$(HOST_CC) $(HOST_CFLAGS) -c -I. -o $@ ../gc/$*.c
