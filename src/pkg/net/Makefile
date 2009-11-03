# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=net
GOFILES=\
	dnsclient.go\
	dnsconfig.go\
	dnsmsg.go\
	fd.go\
	fd_$(GOOS).go\
	ip.go\
	ipsock.go\
	net.go\
	parse.go\
	port.go\
	sock.go\
	tcpsock.go\
	udpsock.go\
	unixsock.go\

include $(GOROOT)/src/Make.pkg
