// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	if a.Zone != "" {
		return JoinHostPort(a.IP.String()+"%"+a.Zone, itoa(a.Port))
	}
	return JoinHostPort(a.IP.String(), itoa(a.Port))
}

// ResolveTCPAddr parses addr as a TCP address of the form "host:port"
// or "[ipv6-host%zone]:port" and resolves a pair of domain name and
// port name on the network net, which must be "tcp", "tcp4" or
// "tcp6".  A literal address or host name for IPv6 must be enclosed
// in square brackets, as in "[::1]:80", "[ipv6-host]:http" or
// "[ipv6-host%zone]:80".
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
