// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP sockets

package net

import (
	"os"
)

// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
	IP   IP
	Port int
}

// Network returns the address's network name, "tcp".
func (a *TCPAddr) Network() string { return "tcp" }

func (a *TCPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return JoinHostPort(a.IP.String(), itoa(a.Port))
}

// ResolveTCPAddr parses addr as a TCP address of the form
// host:port and resolves domain names or port names to
// numeric addresses on the network net, which must be "tcp",
// "tcp4" or "tcp6".  A literal IPv6 host address must be
// enclosed in square brackets, as in "[::]:80".
func ResolveTCPAddr(net, addr string) (*TCPAddr, os.Error) {
	ip, port, err := hostPortToIP(net, addr)
	if err != nil {
		return nil, err
	}
	return &TCPAddr{ip, port}, nil
}
