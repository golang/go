# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=http
GOFILES=\
	chunked.go\
	client.go\
	cookie.go\
	dump.go\
	filetransport.go\
	fs.go\
	header.go\
	lex.go\
	request.go\
	response.go\
	server.go\
	sniff.go\
	status.go\
	transfer.go\
	transport.go\

GOFILES_windows=\
	transport_windows.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../../Make.pkg
