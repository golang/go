# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=gofix
GOFILES=\
	fix.go\
	httpfinalurl.go\
	httpheaders.go\
	httpserver.go\
	main.go\
	netdial.go\
	oserrorstring.go\
	osopen.go\
	procattr.go\
	reflect.go\
	sortslice.go\
	typecheck.go\

include ../../Make.cmd

test:
	gotest

testshort:
	gotest -test.short
