# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=net
GOFILES=\
	dial.go\
	dnsclient.go\
	dnsconfig.go\
	dnsmsg.go\
	fd_$(GOOS).go\
	hosts.go\
	ip.go\
	ipsock.go\
	iprawsock.go\
	net.go\
	parse.go\
	pipe.go\
	port.go\
	sock.go\
	tcpsock.go\
	udpsock.go\
	unixsock.go\

GOFILES_freebsd=\
	newpollserver.go\
	fd.go\

GOFILES_darwin=\
	newpollserver.go\
	fd.go\

GOFILES_linux=\
	newpollserver.go\
	fd.go\

GOFILES_nacl=\
	newpollserver.go\
	fd.go\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
