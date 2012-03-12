# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.dist

install: y.tab.h builtin.c

y.tab.h: go.y go.errors bisonerrors
	bison -v -y -d go.y
	# make yystate global, yytname mutable
	cat y.tab.c | sed '/ int yystate;/d; s/int yychar;/int yychar, yystate;/; s/static const char \*const yytname/const char *yytname/; s/char const \*yymsgp/char *yymsgp/' >y1.tab.c
	mv y1.tab.c y.tab.c
	awk -f bisonerrors y.output go.errors >yerr.h

builtin.c: runtime.go unsafe.go
	./mkbuiltin
