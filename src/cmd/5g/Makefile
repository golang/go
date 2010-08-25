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
	list.$O\
	galign.$O\
	gobj.$O\
	ggen.$O\
	gsubr.$O\
	cgen.$O\
	cgen64.$O

LIB=\
	../gc/gc.a\

include ../../Make.ccmd

%.$O: ../gc/%.c
	$(HOST_CC) $(HOST_CFLAGS) -c -I. -o $@ ../gc/$*.c
