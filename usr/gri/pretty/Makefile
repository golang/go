# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

all: untab godoc pretty

untab: untab.$O
	$(LD) -o untab untab.$O

godoc: godoc.$O
	$(LD) -o godoc godoc.$O

pretty: pretty.$O
	$(LD) -o pretty pretty.$O

test: pretty
	./test.sh

smoketest: pretty
	./test.sh astprinter.go

install: pretty godoc untab
	cp godoc $(HOME)/bin/godoc
	cp pretty $(HOME)/bin/pretty
	cp untab $(HOME)/bin/untab

clean:
	rm -f pretty untab godoc *.$O *.a 6.out *~

godoc.$O:	astprinter.$O

pretty.$O:	 astprinter.$O

%.$O:	%.go
	$(GC) $(F) $<
