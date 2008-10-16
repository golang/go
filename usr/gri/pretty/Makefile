# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

pretty: pretty.6
	$(L) -o pretty pretty.6

test: pretty
	pretty -s *.go
	pretty -s ../gosrc/*.go
	#pretty -s $(GOROOT)/test/*.go  # contains incorrect programs
	pretty -s $(GOROOT)/test/235.go
	pretty -s $(GOROOT)/test/args.go
	pretty -s $(GOROOT)/test/bufiolib.go
	pretty -s $(GOROOT)/test/char_lit.go
	pretty -s $(GOROOT)/test/complit.go
	pretty -s $(GOROOT)/test/const.go
	pretty -s $(GOROOT)/test/dialgoogle.go
	pretty -s $(GOROOT)/test/empty.go
	pretty -s $(GOROOT)/test/env.go
	pretty -s $(GOROOT)/test/float_lit.go
	pretty -s $(GOROOT)/test/fmt_test.go
	pretty -s $(GOROOT)/test/for.go
	pretty -s $(GOROOT)/test/func.go
	pretty -s $(GOROOT)/test/func1.go
	pretty -s $(GOROOT)/test/func2.go
	pretty -s $(GOROOT)/src/pkg/*.go
	pretty -s $(GOROOT)/src/lib/*.go
	pretty -s $(GOROOT)/src/lib/*/*.go
	pretty -s $(GOROOT)/usr/r/*/*.go
	echo "DONE"

testnoisy: pretty
	pretty *.go
	pretty ../gosrc/*.go
	#pretty $(GOROOT)/test/*.go  # contains incorrect programs
	pretty $(GOROOT)/test/235.go
	pretty $(GOROOT)/test/args.go
	pretty $(GOROOT)/test/bufiolib.go
	pretty $(GOROOT)/test/char_lit.go
	pretty $(GOROOT)/test/complit.go
	pretty $(GOROOT)/test/const.go
	pretty $(GOROOT)/test/dialgoogle.go
	pretty $(GOROOT)/test/empty.go
	pretty $(GOROOT)/test/env.go
	pretty $(GOROOT)/test/float_lit.go
	pretty $(GOROOT)/test/fmt_test.go
	pretty $(GOROOT)/test/for.go
	pretty $(GOROOT)/test/func.go
	pretty $(GOROOT)/test/func1.go
	pretty $(GOROOT)/test/func2.go
	pretty $(GOROOT)/src/pkg/*.go
	pretty $(GOROOT)/src/lib/*.go
	pretty $(GOROOT)/src/lib/*/*.go
	pretty $(GOROOT)/usr/r/*/*.go
	echo "DONE"

# These tests don't work yet
testfull: pretty
	pretty *.go
	pretty ../gosrc/*.go
	pretty $(GOROOT)/test/*.go
	pretty $(GOROOT)/src/pkg/*.go
	pretty $(GOROOT)/src/lib/*.go
	pretty $(GOROOT)/src/lib/*/*.go
	pretty $(GOROOT)/usr/r/*/*.go
	echo "DONE"

install: pretty
	cp pretty $(HOME)/bin/pretty

clean:
	rm -f pretty *.6  *~

pretty.6:	parser.6 printer.6 platform.6 scanner.6

printer.6:	node.6 scanner.6

parser.6:	scanner.6 utils.6 printer.6 node.6

node.6:	scanner.6

scanner.6:	utils.6 platform.6

%.6:	%.go
	$(G) $(F) $<
