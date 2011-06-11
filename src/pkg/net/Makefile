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
	interface.go\
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
	dnsclient.go\
	dnsconfig.go\
	fd.go\
	file.go\
	interface_bsd.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock_bsd.go\

CGOFILES_freebsd=\
	cgo_bsd.go\
	cgo_unix.go\

GOFILES_darwin=\
	dnsclient.go\
	dnsconfig.go\
	fd.go\
	file.go\
	interface_bsd.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock_bsd.go\

CGOFILES_darwin=\
	cgo_bsd.go\
	cgo_unix.go\

GOFILES_linux=\
	dnsclient.go\
	dnsconfig.go\
	fd.go\
	file.go\
	interface_linux.go\
	newpollserver.go\
	port.go\
	sendfile_linux.go\
	sock_linux.go\

GOFILES_plan9=\
	interface_stub.go\
	sendfile_stub.go\

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
	file_windows.go\
	interface_stub.go\
	resolv_windows.go\
	sendfile_windows.go\
	sock_windows.go\

GOFILES+=$(GOFILES_$(GOOS))
ifneq ($(CGOFILES_$(GOOS)),)
CGOFILES+=$(CGOFILES_$(GOOS))
endif

include ../../Make.pkg
