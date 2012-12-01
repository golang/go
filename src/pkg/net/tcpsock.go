// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP sockets

package net

// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
	IP   IP
	Port int
	Zone string // IPv6 scoped addressing zone
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
func ResolveTCPAddr(net, addr string) (*TCPAddr, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	case "": // a hint wildcard for Go 1.0 undocumented behavior
		net = "tcp"
	default:
		return nil, UnknownNetworkError(net)
	}
	a, err := resolveInternetAddr(net, addr, noDeadline)
	if err != nil {
		return nil, err
	}
	return a.(*TCPAddr), nil
}
