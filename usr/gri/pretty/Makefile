# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

all: untab gds pretty

untab: untab.6
	$(L) -o untab untab.6

gds: gds.6
	$(L) -o gds gds.6

pretty: pretty.6
	$(L) -o pretty pretty.6

test: pretty
	./test.sh

smoketest: pretty
	./test.sh parser.go

install: pretty
	cp pretty $(HOME)/bin/pretty

clean:
	rm -f pretty *.6 *.a *~

gds.6:	 utils.6 platform.6 compilation.6 printer.6

pretty.6:	 platform.6 printer.6 compilation.6

compilation.6:	 platform.6 parser.6 ast.6 typechecker.6

typechecker.6:	 ast.6

ast.6:	 symboltable.6

symboltable.6:	

parser.6:	ast.6 symboltable.6

platform.6:	 utils.6

printer.6:	 utils.6 ast.6 symboltable.6

%.6:	%.go
	$(G) $(F) $<
