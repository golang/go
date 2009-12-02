# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=unicode
GOFILES=\
	digit.go\
	letter.go\
	tables.go\

include ../../Make.pkg

maketables: maketables.go
	$(GC) maketables.go
	$(LD) -o maketables maketables.$O

tables:	maketables
	$(GC) maketables.go
	$(LD) -o maketables maketables.$O
	./maketables --tables=all > tables.go
	gofmt -w tables.go
	rm -f maketables

testtables: maketables
	echo '***' Be sure to make tables and make install first
	./maketables -test
	rm -f maketables
