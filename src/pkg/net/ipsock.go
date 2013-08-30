// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internet protocol family sockets

package net

import (
	"errors"
	"time"
)

var (
	// supportsIPv4 reports whether the platform supports IPv4
	// networking functionality.
	supportsIPv4 bool

	// supportsIPv6 reports whether the platfrom supports IPv6
	// networking functionality.
	supportsIPv6 bool

	// supportsIPv4map reports whether the platform supports
	// mapping an IPv4 address inside an IPv6 address at transport
	// layer protocols.  See RFC 4291, RFC 4038 and RFC 3493.
	supportsIPv4map bool
)

func init() {
	sysInit()
	supportsIPv4 = probeIPv4Stack()
	supportsIPv6, supportsIPv4map = probeIPv6Stack()
}

// A netaddr represents a network endpoint address or a list of
// network endpoint addresses.
type netaddr interface {
	// toAddr returns the address represented in Addr interface.
	// It returns a nil interface when the address is nil.
	toAddr() Addr
}

var errNoSuitableAddress = errors.New("no suitable address found")

// firstFavoriteAddr returns an address that implemets netaddr
// interface.
func firstFavoriteAddr(filter func(IP) IP, addrs []string, inetaddr func(IP) netaddr) (netaddr, error) {
	if filter == nil {
		// We'll take any IP address, but since the dialing code
		// does not yet try multiple addresses, prefer to use
		// an IPv4 address if possible.  This is especially relevant
		// if localhost resolves to [ipv6-localhost, ipv4-localhost].
		// Too much code assumes localhost == ipv4-localhost.
		addr, err := firstSupportedAddr(ipv4only, addrs, inetaddr)
		if err != nil {
			addr, err = firstSupportedAddr(anyaddr, addrs, inetaddr)
		}
		return addr, err
	} else {
		return firstSupportedAddr(filter, addrs, inetaddr)
	}
}

func firstSupportedAddr(filter func(IP) IP, addrs []string, inetaddr func(IP) netaddr) (netaddr, error) {
	for _, s := range addrs {
		if ip := filter(ParseIP(s)); ip != nil {
			return inetaddr(ip), nil
		}
	}
	return nil, errNoSuitableAddress
}

// anyaddr returns IP addresses that we can use with the current
// kernel configuration.  It returns nil when ip is not suitable for
// the configuration and an IP address.
func anyaddr(ip IP) IP {
	if ip4 := ipv4only(ip); ip4 != nil {
		return ip4
	}
	return ipv6only(ip)
}

// ipv4only returns IPv4 addresses that we can use with the kernel's
// IPv4 addressing modes.  It returns IPv4-mapped IPv6 addresses as
// IPv4 addresses and returns other IPv6 address types as nils.
func ipv4only(ip IP) IP {
	if supportsIPv4 {
		return ip.To4()
	}
	return nil
}

// ipv6only returns IPv6 addresses that we can use with the kernel's
// IPv6 addressing modes.  It returns IPv4-mapped IPv6 addresses as
// nils and returns other IPv6 address types as IPv6 addresses.
func ipv6only(ip IP) IP {
	if supportsIPv6 && len(ip) == IPv6len && ip.To4() == nil {
		return ip
	}
	return nil
}

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

// resolveInternetAddr resolves addr that is either a literal IP
// address or a DNS registered name and returns an internet protocol
// family address.
func resolveInternetAddr(net, addr string, deadline time.Time) (netaddr, error) {
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
	inetaddr := func(ip IP) netaddr {
		switch net {
		case "tcp", "tcp4", "tcp6":
			return &TCPAddr{IP: ip, Port: portnum, Zone: zone}
		case "udp", "udp4", "udp6":
			return &UDPAddr{IP: ip, Port: portnum, Zone: zone}
		case "ip", "ip4", "ip6":
			return &IPAddr{IP: ip, Zone: zone}
		default:
			panic("unexpected network: " + net)
		}
	}
	if host == "" {
		return inetaddr(nil), nil
	}
	// Try as a literal IP address.
	var ip IP
	if ip = parseIPv4(host); ip != nil {
		return inetaddr(ip), nil
	}
	if ip, zone = parseIPv6(host, true); ip != nil {
		return inetaddr(ip), nil
	}
	// Try as a DNS registered name.
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
	return firstFavoriteAddr(filter, addrs, inetaddr)
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
