// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "errors"

var ErrWriteToConnected = errors.New("use of WriteTo with pre-connected UDP")

// UDPAddr represents the address of a UDP end point.
type UDPAddr struct {
	IP   IP
	Port int
	Zone string // IPv6 scoped addressing zone
}

// Network returns the address's network name, "udp".
func (a *UDPAddr) Network() string { return "udp" }

func (a *UDPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	if a.Zone != "" {
		return JoinHostPort(a.IP.String()+"%"+a.Zone, itoa(a.Port))
	}
	return JoinHostPort(a.IP.String(), itoa(a.Port))
}

// ResolveUDPAddr parses addr as a UDP address of the form "host:port"
// or "[ipv6-host%zone]:port" and resolves a pair of domain name and
// port name on the network net, which must be "udp", "udp4" or
// "udp6".  A literal address or host name for IPv6 must be enclosed
// in square brackets, as in "[::1]:80", "[ipv6-host]:http" or
// "[ipv6-host%zone]:80".
func ResolveUDPAddr(net, addr string) (*UDPAddr, error) {
	switch net {
	case "udp", "udp4", "udp6":
	case "": // a hint wildcard for Go 1.0 undocumented behavior
		net = "udp"
	default:
		return nil, UnknownNetworkError(net)
	}
	a, err := resolveInternetAddr(net, addr, noDeadline)
	if err != nil {
		return nil, err
	}
	return a.(*UDPAddr), nil
}
