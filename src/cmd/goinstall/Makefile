# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=goinstall
GOFILES=\
	download.go\
	main.go\
	make.go\
	parse.go\
	path.go\
	syslist.go\

CLEANFILES+=syslist.go

include ../../Make.cmd

syslist.go:
	echo '// Generated automatically by make.' >$@
	echo 'package main' >>$@
	echo 'const goosList = "$(GOOS_LIST)"' >>$@
	echo 'const goarchList = "$(GOARCH_LIST)"' >>$@

test:
	gotest

testshort:
	gotest -test.short
