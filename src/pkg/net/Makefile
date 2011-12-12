# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=net
GOFILES=\
	dial.go\
	dnsclient.go\
	dnsmsg.go\
	hosts.go\
	interface.go\
	ip.go\
	iprawsock.go\
	ipsock.go\
	net.go\
	parse.go\
	pipe.go\
	tcpsock.go\
	udpsock.go\
	unixsock.go\

GOFILES_freebsd=\
	dnsclient_unix.go\
	dnsconfig.go\
	fd.go\
	fd_$(GOOS).go\
	file.go\
	interface_bsd.go\
	interface_freebsd.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_unix.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock.go\
	sock_bsd.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\

ifeq ($(CGO_ENABLED),1)
CGOFILES_freebsd=\
	cgo_bsd.go\
	cgo_unix.go
else
GOFILES_freebsd+=cgo_stub.go
endif

GOFILES_darwin=\
	dnsclient_unix.go\
	dnsconfig.go\
	fd.go\
	fd_$(GOOS).go\
	file.go\
	interface_bsd.go\
	interface_darwin.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_unix.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock.go\
	sock_bsd.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\

ifeq ($(CGO_ENABLED),1)
CGOFILES_darwin=\
	cgo_bsd.go\
	cgo_unix.go
else
GOFILES_darwin+=cgo_stub.go
endif

GOFILES_linux=\
	dnsclient_unix.go\
	dnsconfig.go\
	fd.go\
	fd_$(GOOS).go\
	file.go\
	interface_linux.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_unix.go\
	newpollserver.go\
	port.go\
	sendfile_linux.go\
	sock.go\
	sock_linux.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\

ifeq ($(CGO_ENABLED),1)
CGOFILES_linux=\
	cgo_linux.go\
	cgo_unix.go
else
GOFILES_linux+=cgo_stub.go
endif

GOFILES_netbsd=\
	dnsclient_unix.go\
	dnsconfig.go\
	fd.go\
	fd_$(GOOS).go\
	file.go\
	interface_bsd.go\
	interface_openbsd.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_unix.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock.go\
	sock_bsd.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\
	cgo_stub.go\

GOFILES_openbsd=\
	dnsclient_unix.go\
	dnsconfig.go\
	fd.go\
	fd_$(GOOS).go\
	file.go\
	interface_bsd.go\
	interface_openbsd.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_unix.go\
	newpollserver.go\
	port.go\
	sendfile_stub.go\
	sock.go\
	sock_bsd.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\
	cgo_stub.go\

GOFILES_plan9=\
	file_plan9.go\
	interface_stub.go\
	iprawsock_plan9.go\
	ipsock_plan9.go\
	lookup_plan9.go\
	tcpsock_plan9.go\
	udpsock_plan9.go\
	unixsock_plan9.go\

GOFILES_windows=\
	fd_$(GOOS).go\
	file_windows.go\
	interface_windows.go\
	iprawsock_posix.go\
	ipsock_posix.go\
	lookup_windows.go\
	sendfile_windows.go\
	sock.go\
	sock_windows.go\
	tcpsock_posix.go\
	udpsock_posix.go\
	unixsock_posix.go\

GOFILES+=$(GOFILES_$(GOOS))
ifneq ($(CGOFILES_$(GOOS)),)
CGOFILES+=$(CGOFILES_$(GOOS))
endif

include ../../Make.pkg
