# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

pretty: pretty.6
	$(L) -o pretty pretty.6

test: pretty
	test.sh

install: pretty
	cp pretty $(HOME)/bin/pretty

clean:
	rm -f pretty *.6  *~

pretty.6:	printer.6 platform.6 compilation.6

compilation.6:	scanner.6 parser.6 ast.6

printer.6:	ast.6 scanner.6

parser.6:	scanner.6 utils.6 printer.6 ast.6

ast.6:	scanner.6

scanner.6:	utils.6 platform.6

%.6:	%.go
	$(G) $(F) $<
