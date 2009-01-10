# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

all: untab pretty

untab: untab.6
	$(L) -o untab untab.6
	
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

pretty.6:	 platform.6 printer.6 compilation.6

compilation.6:	 platform.6 scanner.6 parser.6 ast.6 typechecker.6

ast.6:	 scanner.6

scanner.6:	 utils.6

parser.6:	 scanner.6 ast.6

platform.6:	 utils.6

printer.6:	 scanner.6 ast.6 utils.6

typechecker.6:	ast.6 universe.6

universe.6:	ast.6

%.6:	%.go
	$(G) $(F) $<
