# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

ALL=\
	parser\
	peano\
	tree\
	tree2\

all: $(ALL)

%: %.go
	go build $*.go stats.go

%.bench: %
	time ./$*

bench: $(addsuffix .bench, $(ALL))

clean:
	rm -f $(ALL)

