# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

TARG=5l

OFILES=\
	asm.$O\
	data.$O\
	elf.$O\
	enam.$O\
	ldelf.$O\
	ldmacho.$O\
	ldpe.$O\
	lib.$O\
	list.$O\
	noop.$O\
	obj.$O\
	optab.$O\
	pass.$O\
	prof.$O\
	softfloat.$O\
	span.$O\
	symtab.$O\
	go.$O\

HFILES=\
	l.h\
	5.out.h\
	../ld/elf.h\

include ../../Make.ctool

enam.c: 5.out.h
	sh mkenam

CLEANFILES+=enam.c

%.$O: ../ld/%.c
	$(HOST_CC) $(HOST_CFLAGS) -c -I. ../ld/$*.c
