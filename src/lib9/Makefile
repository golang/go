# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../Make.inc
O:=$(HOST_O)

LIB=lib9.a

NUM=\
	charstod.$O\
	pow10.$O\

# Could add fmt/errfmt, but we want to pick it up from ./errstr.c instead.
FMTOFILES=\
	dofmt.$O\
	fltfmt.$O\
	fmt.$O\
	fmtfd.$O\
	fmtfdflush.$O\
	fmtlocale.$O\
	fmtlock2.$O\
	fmtnull.$O\
	fmtprint.$O\
	fmtquote.$O\
	fmtrune.$O\
	fmtstr.$O\
	fmtvprint.$O\
	fprint.$O\
	nan64.$O\
	print.$O\
	seprint.$O\
	smprint.$O\
	snprint.$O\
	sprint.$O\
	strtod.$O\
	vfprint.$O\
	vseprint.$O\
	vsmprint.$O\
	vsnprint.$O\
	$(NUM)\

UTFOFILES=\
	rune.$O\
	utfecpy.$O\
	utflen.$O\
	utfnlen.$O\
	utfrrune.$O\
	utfrune.$O\
	utfutf.$O\
	runetype.$O\

LIB9OFILES=\
	_p9dir.$O\
	_exits.$O\
	argv0.$O\
	atoi.$O\
	cleanname.$O\
	create.$O\
	ctime.$O\
	dirfstat.$O\
	dirfwstat.$O\
	dirstat.$O\
	dirwstat.$O\
	dup.$O\
	errstr.$O\
	exec.$O\
	execl.$O\
	exitcode.$O\
	exits.$O\
	getenv.$O\
	getfields.$O\
	getwd.$O\
	goos.$O\
	main.$O\
	nan.$O\
	nulldir.$O\
	open.$O\
	readn.$O\
	seek.$O\
	strecpy.$O\
	sysfatal.$O\
	time.$O\
	tokenize.$O\

ifeq ($(GOHOSTOS),windows)
LIB9OFILES+=\
	win32.$O\

else
LIB9OFILES+=\
	await.$O\
	getuser.$O\
	jmp.$O\
	notify.$O\
	rfork.$O\

endif

OFILES=\
	$(LIB9OFILES)\
	$(FMTOFILES)\
	$(UTFOFILES)\

HFILES=\
	$(QUOTED_GOROOT)/include/u.h\
	$(QUOTED_GOROOT)/include/libc.h\

include ../Make.clib

GOROOT_FINAL?=$(GOROOT)

%.$O: fmt/%.c
	$(HOST_CC) -c $(HOST_CFLAGS) -DPLAN9PORT -Ifmt $<

%.$O: utf/%.c
	$(HOST_CC) -c $(HOST_CFLAGS) $<

goos.$O: goos.c
	GOVERSION=`../version.bash` && \
		$(HOST_CC) -c $(HOST_CFLAGS) -DGOOS='"$(GOOS)"' -DGOARCH='"$(GOARCH)"' -DGOROOT='"$(GOROOT_FINAL)"' -DGOVERSION='"'"$$GOVERSION"'"' $<

