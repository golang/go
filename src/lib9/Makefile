# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../Make.conf

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

ifeq ($(GOOS),mingw)
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
	"$(GOROOT)"/include/u.h\
	"$(GOROOT)"/include/libc.h\

install: $(LIB)
	cp $(LIB) "$(GOROOT)/lib"

$(LIB): $(OFILES)
	ar rsc $(LIB) $(OFILES)

%.$O: fmt/%.c
	$(CC) -c $(CFLAGS) -DPLAN9PORT -Ifmt $<

%.$O: utf/%.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *.$O *.6 6.out $(LIB)

nuke: clean
	rm -f "$(GOROOT)"/lib/$(LIB)

#XLIB=$PLAN9/lib/$LIB

#testfmt: testfmt.$O $XLIB
#	$LD -o $target testfmt.$O

#testfltfmt: testfltfmt.$O $XLIB
#	$LD -o $target testfltfmt.$O

#testprint: testprint.$O $XLIB
#	$LD -o $target testprint.$O

