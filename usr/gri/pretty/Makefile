# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

pretty: pretty.6
	$(L) -o pretty pretty.6

test: all
	pretty *.go
	# pretty $(GOROOT)/test/fixedbugs/*.go  # some files legally don't compile
	pretty $(GOROOT)/test/sieve.go
	pretty $(GOROOT)/src/pkg/*.go
	pretty $(GOROOT)/src/lib/flag.go
	pretty $(GOROOT)/src/lib/fmt.go
	pretty $(GOROOT)/src/lib/rand.go
	pretty $(GOROOT)/src/lib/math/*.go
	pretty $(GOROOT)/src/lib/container/*.go
	pretty $(GOROOT)/src/syscall/*.go
	pretty base.go decls.go
	pretty -token_chan base.go decls.go
	echo "PASSED"

install: pretty
	cp pretty $(HOME)/bin/pretty

clean:
	rm -f pretty *.6  *~

pretty.6:	parser.6 printer.6 platform.6 scanner.6

parser.6:	ast.6 scanner.6 utils.6

scanner.6:	utils.6 platform.6

%.6:	%.go
	$(G) $(F) $<
