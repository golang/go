// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internet protocol family sockets

package net

import "time"

var supportsIPv6, supportsIPv4map bool

func init() {
	sysInit()
	supportsIPv6, supportsIPv4map = probeIPv6Stack()
}

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
	host, port, _, err = splitHostPort(hostport)
	return
}

func splitHostPort(hostport string) (host, port, zone string, err error) {
	// The port starts after the last colon.
	i := last(hostport, ':')
	if i < 0 {
		err = &AddrError{"missing port in address", hostport}
		return
	}
	host, port = hostport[:i], hostport[i+1:]
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

func resolveInternetAddr(net, addr string, deadline time.Time) (Addr, error) {
	var (
		err              error
		host, port, zone string
		portnum          int
	)
	switch net {
	case "tcp", "tcp4", "tcp6", "udp", "udp4", "udp6":
		if addr != "" {
			if host, port, zone, err = splitHostPort(addr); err != nil {
				return nil, err
			}
			if portnum, err = parsePort(net, port); err != nil {
				return nil, err
			}
		}
	case "ip", "ip4", "ip6":
		if addr != "" {
			host = addr
		}
	default:
		return nil, UnknownNetworkError(net)
	}
	inetaddr := func(net string, ip IP, port int, zone string) Addr {
		switch net {
		case "tcp", "tcp4", "tcp6":
			return &TCPAddr{IP: ip, Port: port, Zone: zone}
		case "udp", "udp4", "udp6":
			return &UDPAddr{IP: ip, Port: port, Zone: zone}
		case "ip", "ip4", "ip6":
			return &IPAddr{IP: ip, Zone: zone}
		}
		return nil
	}
	if host == "" {
		return inetaddr(net, nil, portnum, zone), nil
	}
	// Try as an IP address.
	if ip := ParseIP(host); ip != nil {
		return inetaddr(net, ip, portnum, zone), nil
	}
	var filter func(IP) IP
	if net != "" && net[len(net)-1] == '4' {
		filter = ipv4only
	}
	if net != "" && net[len(net)-1] == '6' {
		filter = ipv6only
	}
	// Try as a DNS name.
	addrs, err := lookupHostDeadline(host, deadline)
	if err != nil {
		return nil, err
	}
	ip := firstFavoriteAddr(filter, addrs)
	if ip == nil {
		// should not happen
		return nil, &AddrError{"LookupHost returned no suitable address", addrs[0]}
	}
	return inetaddr(net, ip, portnum, zone), nil
}

func zoneToString(zone int) string {
	if zone == 0 {
		return ""
	}
	if ifi, err := InterfaceByIndex(zone); err == nil {
		return ifi.Name
	}
	return itod(uint(zone))
}

func zoneToInt(zone string) int {
	if zone == "" {
		return 0
	}
	if ifi, err := InterfaceByName(zone); err == nil {
		return ifi.Index
	}
	n, _, _ := dtoi(zone, 0)
	return n
}
