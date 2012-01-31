// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP sockets

package net

var supportsIPv6, supportsIPv4map = probeIPv6Stack()

func firstFavoriteAddr(filter func(IP) IP, addrs []string) (addr IP) {
	if filter == nil {
		// We'll take any IP address, but since the dialing code
		// does not yet try multiple addresses, prefer to use
		// an IPv4 address if possible.  This is especially relevant
		// if localhost resolves to [ipv6-localhost, ipv4-localhost].
		// Too much code assumes localhost == ipv4-localhost.
		addr = firstSupportedAddr(ipv4only, addrs)
		if addr == nil {
			addr = firstSupportedAddr(anyaddr, addrs)
		}
	} else {
		addr = firstSupportedAddr(filter, addrs)
	}
	return
}

func firstSupportedAddr(filter func(IP) IP, addrs []string) IP {
	for _, s := range addrs {
		if addr := filter(ParseIP(s)); addr != nil {
			return addr
		}
	}
	return nil
}

func anyaddr(x IP) IP {
	if x4 := x.To4(); x4 != nil {
		return x4
	}
	if supportsIPv6 {
		return x
	}
	return nil
}

func ipv4only(x IP) IP { return x.To4() }

func ipv6only(x IP) IP {
	// Only return addresses that we can use
	// with the kernel's IPv6 addressing modes.
	if len(x) == IPv6len && x.To4() == nil && supportsIPv6 {
		return x
	}
	return nil
}

type InvalidAddrError string

func (e InvalidAddrError) Error() string   { return string(e) }
func (e InvalidAddrError) Timeout() bool   { return false }
func (e InvalidAddrError) Temporary() bool { return false }

// SplitHostPort splits a network address of the form
// "host:port" or "[host]:port" into host and port.
// The latter form must be used when host contains a colon.
func SplitHostPort(hostport string) (host, port string, err error) {
	// The port starts after the last colon.
	i := last(hostport, ':')
	if i < 0 {
		err = &AddrError{"missing port in address", hostport}
		return
	}

	host, port = hostport[0:i], hostport[i+1:]

	// Can put brackets around host ...
	if len(host) > 0 && host[0] == '[' && host[len(host)-1] == ']' {
		host = host[1 : len(host)-1]
	} else {
		// ... but if there are no brackets, no colons.
		if byteIndex(host, ':') >= 0 {
			err = &AddrError{"too many colons in address", hostport}
			return
		}
	}
	return
}

// JoinHostPort combines host and port into a network address
// of the form "host:port" or, if host contains a colon, "[host]:port".
func JoinHostPort(host, port string) string {
	// If host has colons, have to bracket it.
	if byteIndex(host, ':') >= 0 {
		return "[" + host + "]:" + port
	}
	return host + ":" + port
}

// Convert "host:port" into IP address and port.
func hostPortToIP(net, hostport string) (ip IP, iport int, err error) {
	host, port, err := SplitHostPort(hostport)
	if err != nil {
		return nil, 0, err
	}

	var addr IP
	if host != "" {
		// Try as an IP address.
		addr = ParseIP(host)
		if addr == nil {
			var filter func(IP) IP
			if net != "" && net[len(net)-1] == '4' {
				filter = ipv4only
			}
			if net != "" && net[len(net)-1] == '6' {
				filter = ipv6only
			}
			// Not an IP address.  Try as a DNS name.
			addrs, err := LookupHost(host)
			if err != nil {
				return nil, 0, err
			}
			addr = firstFavoriteAddr(filter, addrs)
			if addr == nil {
				// should not happen
				return nil, 0, &AddrError{"LookupHost returned no suitable address", addrs[0]}
			}
		}
	}

	p, i, ok := dtoi(port, 0)
	if !ok || i != len(port) {
		p, err = LookupPort(net, port)
		if err != nil {
			return nil, 0, err
		}
	}
	if p < 0 || p > 0xFFFF {
		return nil, 0, &AddrError{"invalid port", port}
	}

	return addr, p, nil

}
