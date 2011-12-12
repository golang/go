# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=crypto/tls
GOFILES=\
	alert.go\
	cipher_suites.go\
	common.go\
	conn.go\
	handshake_client.go\
	handshake_messages.go\
	handshake_server.go\
	key_agreement.go\
	prf.go\
	tls.go\

ifeq ($(CGO_ENABLED),1)
CGOFILES_darwin=\
	root_darwin.go
else
GOFILES_darwin+=root_stub.go
endif

GOFILES_freebsd+=root_unix.go
GOFILES_linux+=root_unix.go
GOFILES_netbsd+=root_unix.go
GOFILES_openbsd+=root_unix.go
GOFILES_plan9+=root_stub.go
GOFILES_windows+=root_windows.go

GOFILES+=$(GOFILES_$(GOOS))
ifneq ($(CGOFILES_$(GOOS)),)
CGOFILES+=$(CGOFILES_$(GOOS))
endif

include ../../../Make.pkg
