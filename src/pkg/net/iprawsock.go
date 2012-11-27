// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Raw IP sockets

package net

// IPAddr represents the address of an IP end point.
type IPAddr struct {
	IP   IP
	Zone string // IPv6 scoped addressing zone
}

// Network returns the address's network name, "ip".
func (a *IPAddr) Network() string { return "ip" }

func (a *IPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return a.IP.String()
}

// ResolveIPAddr parses addr as an IP address and resolves domain
// names to numeric addresses on the network net, which must be
// "ip", "ip4" or "ip6".  A literal IPv6 host address must be
// enclosed in square brackets, as in "[::]".
func ResolveIPAddr(net, addr string) (*IPAddr, error) {
	afnet, _, err := parseDialNetwork(net)
	if err != nil {
		return nil, err
	}
	switch afnet {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(net)
	}
	a, err := resolveInternetAddr(afnet, addr, noDeadline)
	if err != nil {
		return nil, err
	}
	return a.(*IPAddr), nil
}
