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

// SplitHostPort splits a network address of the form "host:port",
// "[host]:port" or "[ipv6-host%zone]:port" into host or
// ipv6-host%zone and port.  A literal address or host name for IPv6
// must be enclosed in square brackets, as in "[::1]:80",
// "[ipv6-host]:http" or "[ipv6-host%zone]:80".
func SplitHostPort(hostport string) (host, port string, err error) {
	j, k := 0, 0

	// The port starts after the last colon.
	i := last(hostport, ':')
	if i < 0 {
		goto missingPort
	}

	if hostport[0] == '[' {
		// Expect the first ']' just before the last ':'.
		end := byteIndex(hostport, ']')
		if end < 0 {
			err = &AddrError{"missing ']' in address", hostport}
			return
		}
		switch end + 1 {
		case len(hostport):
			// There can't be a ':' behind the ']' now.
			goto missingPort
		case i:
			// The expected result.
		default:
			// Either ']' isn't followed by a colon, or it is
			// followed by a colon that is not the last one.
			if hostport[end+1] == ':' {
				goto tooManyColons
			}
			goto missingPort
		}
		host = hostport[1:end]
		j, k = 1, end+1 // there can't be a '[' resp. ']' before these positions
	} else {
		host = hostport[:i]
		if byteIndex(host, ':') >= 0 {
			goto tooManyColons
		}
		if byteIndex(host, '%') >= 0 {
			goto missingBrackets
		}
	}
	if byteIndex(hostport[j:], '[') >= 0 {
		err = &AddrError{"unexpected '[' in address", hostport}
		return
	}
	if byteIndex(hostport[k:], ']') >= 0 {
		err = &AddrError{"unexpected ']' in address", hostport}
		return
	}

	port = hostport[i+1:]
	return

missingPort:
	err = &AddrError{"missing port in address", hostport}
	return

tooManyColons:
	err = &AddrError{"too many colons in address", hostport}
	return

missingBrackets:
	err = &AddrError{"missing brackets in address", hostport}
	return
}

func splitHostZone(s string) (host, zone string) {
	// The IPv6 scoped addressing zone identifer starts after the
	// last percent sign.
	if i := last(s, '%'); i > 0 {
		host, zone = s[:i], s[i+1:]
	} else {
		host = s
	}
	return
}

// JoinHostPort combines host and port into a network address of the
// form "host:port" or, if host contains a colon or a percent sign,
// "[host]:port".
func JoinHostPort(host, port string) string {
	// If host has colons or a percent sign, have to bracket it.
	if byteIndex(host, ':') >= 0 || byteIndex(host, '%') >= 0 {
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
			if host, port, err = SplitHostPort(addr); err != nil {
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
	if ip := parseIPv4(host); ip != nil {
		return inetaddr(net, ip, portnum, zone), nil
	}
	if ip, zone := parseIPv6(host, true); ip != nil {
		return inetaddr(net, ip, portnum, zone), nil
	}
	// Try as a domain name.
	host, zone = splitHostZone(host)
	addrs, err := lookupHostDeadline(host, deadline)
	if err != nil {
		return nil, err
	}
	var filter func(IP) IP
	if net != "" && net[len(net)-1] == '4' {
		filter = ipv4only
	}
	if net != "" && net[len(net)-1] == '6' || zone != "" {
		filter = ipv6only
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
