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
	dnsconfig.go\
	dnsclient.go\
	port.go\

GOFILES_darwin=\
	newpollserver.go\
	fd.go\
	dnsconfig.go\
	dnsclient.go\
	port.go\
	
GOFILES_linux=\
	newpollserver.go\
	fd.go\
	dnsconfig.go\
	dnsclient.go\
	port.go\

GOFILES_windows=\
	resolv_windows.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
