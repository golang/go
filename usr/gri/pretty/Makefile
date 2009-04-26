# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

all: untab godoc pretty

untab: untab.6
	$(L) -o untab untab.6

godoc: godoc.6
	$(L) -o godoc godoc.6

pretty: pretty.6
	$(L) -o pretty pretty.6

test: pretty
	./test.sh

smoketest: pretty
	./test.sh astprinter.go

install: pretty godoc untab
	cp godoc $(HOME)/bin/godoc
	cp pretty $(HOME)/bin/pretty
	cp untab $(HOME)/bin/untab

clean:
	rm -f pretty untab godoc *.6 *.a 6.out *~

godoc.6:	astprinter.6 comment.6

pretty.6:	 astprinter.6 format.6

%.6:	%.go
	$(G) $(F) $<
