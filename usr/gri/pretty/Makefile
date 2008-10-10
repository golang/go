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
	pretty -s $(GOROOT)/test/*.go
	pretty -s $(GOROOT)/src/pkg/*.go
	pretty -s $(GOROOT)/src/lib/*.go
	pretty -s $(GOROOT)/src/lib/*/*.go
	pretty -s $(GOROOT)/usr/r/*/*.go
	echo "DONE"

testnoisy: pretty
	pretty *.go
	pretty ../gosrc/*.go
	pretty $(GOROOT)/test/235.go
	pretty $(GOROOT)/test/args.go
	pretty $(GOROOT)/test/bufiolib.go
	pretty $(GOROOT)/test/char_lit.go
	pretty $(GOROOT)/test/sieve.go
	pretty $(GOROOT)/src/pkg/*.go
	pretty $(GOROOT)/src/lib/flag.go
	pretty $(GOROOT)/src/lib/fmt.go
	pretty $(GOROOT)/src/lib/rand.go
	pretty $(GOROOT)/src/lib/math/*.go
	pretty $(GOROOT)/src/lib/container/*.go
	pretty $(GOROOT)/src/lib/syscall/*.go
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

parser.6:	ast.6 scanner.6 utils.6 printer.6

scanner.6:	utils.6 platform.6

%.6:	%.go
	$(G) $(F) $<
