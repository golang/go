# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=gofmt
GOFILES=\
	gofmt.go\
	rewrite.go\
	simplify.go\

include ../../Make.cmd

test: $(TARG)
	./test.sh

testshort:
	gotest -test.short
