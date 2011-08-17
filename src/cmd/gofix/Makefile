# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=gofix
GOFILES=\
	filepath.go\
	fix.go\
	httpfinalurl.go\
	httpfs.go\
	httpheaders.go\
	httpserver.go\
	main.go\
	netdial.go\
	oserrorstring.go\
	osopen.go\
	procattr.go\
	reflect.go\
	signal.go\
	sorthelpers.go\
	sortslice.go\
	stringssplit.go\
	typecheck.go\
	url.go\

include ../../Make.cmd

test:
	gotest

testshort:
	gotest -test.short
