# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../src/Make.$(GOARCH)

ALL=\
	parser\
	peano\
	tree\

all: $(ALL)

%.$O: %.go
	$(GC) $*.go

%: %.$O
	$(LD) -o $@ $*.$O

%.bench: %
	./$*

bench: $(addsuffix .bench, $(ALL))

clean:
	rm -f *.[$(OS)] $(ALL)

