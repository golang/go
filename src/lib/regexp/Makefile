# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

A=6
G=$(A)g
L=$(A)l
PKG=$(GOROOT)/pkg/regexp.$A

test:	main.$A test.$A
	$L -o test test.$A
	./test

install:	regexp.$A
	cp regexp.$A $(PKG)

main:	main.$A
	$L -o main main.$A

main.$A: regexp.$A

clean:
	rm -f *.6 test

%.6:	%.go
	$G $<
