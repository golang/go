# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=gofix
GOFILES=\
	error.go\
	filepath.go\
	fix.go\
	go1pkgrename.go\
	googlecode.go\
	hashsum.go\
	htmlerr.go\
	httpfinalurl.go\
	httpfs.go\
	httpheaders.go\
	httpserver.go\
	httputil.go\
	imagecolor.go\
	imagenew.go\
	iocopyn.go\
	main.go\
	mapdelete.go\
	math.go\
	netdial.go\
	netudpgroup.go\
	oserrorstring.go\
	osopen.go\
	procattr.go\
	reflect.go\
	signal.go\
	sorthelpers.go\
	sortslice.go\
	strconv.go\
	stringssplit.go\
	template.go\
	timefileinfo.go\
	typecheck.go\
	url.go\

include ../../Make.cmd

test:
	gotest

testshort:
	gotest -test.short
