// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

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
	ip := ipEmptyString(a.IP)
	if a.Zone != "" {
		return JoinHostPort(ip+"%"+a.Zone, itoa(a.Port))
	}
	return JoinHostPort(ip, itoa(a.Port))
}

func (a *UDPAddr) isWildcard() bool {
	if a == nil || a.IP == nil {
		return true
	}
	return a.IP.IsUnspecified()
}

func (a *UDPAddr) opAddr() Addr {
	if a == nil {
		return nil
	}
	return a
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
	addrs, err := internetAddrList(net, addr, noDeadline)
	if err != nil {
		return nil, err
	}
	return addrs.first(isIPv4).(*UDPAddr), nil
}
