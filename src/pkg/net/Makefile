# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=net
GOFILES=\
	dial.go\
	dnsmsg.go\
	fd_$(GOOS).go\
	hosts.go\
	ip.go\
	ipsock.go\
	iprawsock.go\
	lookup.go\
	net.go\
	parse.go\
	pipe.go\
	sock.go\
	tcpsock.go\
	udpsock.go\
	unixsock.go\

GOFILES_freebsd=\
	newpollserver.go\
	fd.go\
	file.go\
	dnsconfig.go\
	dnsclient.go\
	port.go\

CGOFILES_freebsd=\
	cgo_bsd.go\
	cgo_unix.go\

GOFILES_darwin=\
	newpollserver.go\
	fd.go\
	file.go\
	dnsconfig.go\
	dnsclient.go\
	port.go\

CGOFILES_darwin=\
	cgo_bsd.go\
	cgo_unix.go\
	
GOFILES_linux=\
	newpollserver.go\
	fd.go\
	file.go\
	dnsconfig.go\
	dnsclient.go\
	port.go\

ifeq ($(GOARCH),arm)
# ARM has no cgo, so use the stubs.
GOFILES_linux+=cgo_stub.go
else
CGOFILES_linux=\
	cgo_linux.go\
	cgo_unix.go
endif

GOFILES_windows=\
	cgo_stub.go\
	resolv_windows.go\
	file_windows.go\

GOFILES+=$(GOFILES_$(GOOS))
ifneq ($(CGOFILES_$(GOOS)),)
CGOFILES+=$(CGOFILES_$(GOOS))
endif

include ../../Make.pkg
