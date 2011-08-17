// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// (Raw) IP sockets

package net

import (
	"os"
)

// IPAddr represents the address of a IP end point.
type IPAddr struct {
	IP IP
}

// Network returns the address's network name, "ip".
func (a *IPAddr) Network() string { return "ip" }

func (a *IPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return a.IP.String()
}

// ResolveIPAddr parses addr as a IP address and resolves domain
// names to numeric addresses on the network net, which must be
// "ip", "ip4" or "ip6".  A literal IPv6 host address must be
// enclosed in square brackets, as in "[::]".
func ResolveIPAddr(net, addr string) (*IPAddr, os.Error) {
	ip, err := hostToIP(net, addr)
	if err != nil {
		return nil, err
	}
	return &IPAddr{ip}, nil
}

// Convert "host" into IP address.
func hostToIP(net, host string) (ip IP, err os.Error) {
	var addr IP
	// Try as an IP address.
	addr = ParseIP(host)
	if addr == nil {
		filter := anyaddr
		if net != "" && net[len(net)-1] == '4' {
			filter = ipv4only
		}
		if net != "" && net[len(net)-1] == '6' {
			filter = ipv6only
		}
		// Not an IP address.  Try as a DNS name.
		addrs, err1 := LookupHost(host)
		if err1 != nil {
			err = err1
			goto Error
		}
		addr = firstFavoriteAddr(filter, addrs)
		if addr == nil {
			// should not happen
			err = &AddrError{"LookupHost returned no suitable address", addrs[0]}
			goto Error
		}
	}
	return addr, nil
Error:
	return nil, err
}
