# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CC=6c -w
GC=6g
LD=6l
O=6

TARG=testrandom testrepeat testsizetoclass

default: $(TARG)

%.$O: %.c malloc.h
	$(CC) $*.c

%.$O: %.go
	$(GC) $*.go

OFILES=\
	allocator.$O\
	malloc.$O\
	mem.$O\
	ms.$O\
	pagemap.$O\
	triv.$O\

testrandom.$O: allocator.$O
testrepeat.$O: allocator.$O

test%: test%.$O $(OFILES)
	$(LD) -o $@ $^

clean:
	rm -f *.$O $(TARG)

runtime: $(OFILES)
	6ar grc $(GOROOT)/lib/lib_$(GOARCH)_$(GOOS).a $^
