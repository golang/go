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

	// supportsIPv6 reports whether the platform supports IPv6
	// networking functionality.
	supportsIPv6 bool

	// supportsIPv4map reports whether the platform supports
	// mapping an IPv4 address inside an IPv6 address at transport
	// layer protocols.  See RFC 4291, RFC 4038 and RFC 3493.
	supportsIPv4map bool
)

// An addrList represents a list of network endpoint addresses.
type addrList []Addr

// isIPv4 returns true if the Addr contains an IPv4 address.
func isIPv4(addr Addr) bool {
	switch addr := addr.(type) {
	case *TCPAddr:
		return addr.IP.To4() != nil
	case *UDPAddr:
		return addr.IP.To4() != nil
	case *IPAddr:
		return addr.IP.To4() != nil
	}
	return false
}

// first returns the first address which satisfies strategy, or if
// none do, then the first address of any kind.
func (addrs addrList) first(strategy func(Addr) bool) Addr {
	for _, addr := range addrs {
		if strategy(addr) {
			return addr
		}
	}
	return addrs[0]
}

// partition divides an address list into two categories, using a
// strategy function to assign a boolean label to each address.
// The first address, and any with a matching label, are returned as
// primaries, while addresses with the opposite label are returned
// as fallbacks. For non-empty inputs, primaries is guaranteed to be
// non-empty.
func (addrs addrList) partition(strategy func(Addr) bool) (primaries, fallbacks addrList) {
	var primaryLabel bool
	for i, addr := range addrs {
		label := strategy(addr)
		if i == 0 || label == primaryLabel {
			primaryLabel = label
			primaries = append(primaries, addr)
		} else {
			fallbacks = append(fallbacks, addr)
		}
	}
	return
}

var errNoSuitableAddress = errors.New("no suitable address found")

// filterAddrList applies a filter to a list of IP addresses,
// yielding a list of Addr objects. Known filters are nil, ipv4only,
// and ipv6only. It returns every address when the filter is nil.
// The result contains at least one address when error is nil.
func filterAddrList(filter func(IPAddr) bool, ips []IPAddr, inetaddr func(IPAddr) Addr) (addrList, error) {
	var addrs addrList
	for _, ip := range ips {
		if filter == nil || filter(ip) {
			addrs = append(addrs, inetaddr(ip))
		}
	}
	if len(addrs) == 0 {
		return nil, errNoSuitableAddress
	}
	return addrs, nil
}

// ipv4only reports whether the kernel supports IPv4 addressing mode
// and addr is an IPv4 address.
func ipv4only(addr IPAddr) bool {
	return supportsIPv4 && addr.IP.To4() != nil
}

// ipv6only reports whether the kernel supports IPv6 addressing mode
// and addr is an IPv6 address except IPv4-mapped IPv6 address.
func ipv6only(addr IPAddr) bool {
	return supportsIPv6 && len(addr.IP) == IPv6len && addr.IP.To4() == nil
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
			err = &AddrError{Err: "missing ']' in address", Addr: hostport}
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
		err = &AddrError{Err: "unexpected '[' in address", Addr: hostport}
		return
	}
	if byteIndex(hostport[k:], ']') >= 0 {
		err = &AddrError{Err: "unexpected ']' in address", Addr: hostport}
		return
	}

	port = hostport[i+1:]
	return

missingPort:
	err = &AddrError{Err: "missing port in address", Addr: hostport}
	return

tooManyColons:
	err = &AddrError{Err: "too many colons in address", Addr: hostport}
	return

missingBrackets:
	err = &AddrError{Err: "missing brackets in address", Addr: hostport}
	return
}

func splitHostZone(s string) (host, zone string) {
	// The IPv6 scoped addressing zone identifier starts after the
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

// internetAddrList resolves addr, which may be a literal IP
// address or a DNS name, and returns a list of internet protocol
// family addresses. The result contains at least one address when
// error is nil.
func internetAddrList(net, addr string, deadline time.Time) (addrList, error) {
	var (
		err        error
		host, port string
		portnum    int
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
	inetaddr := func(ip IPAddr) Addr {
		switch net {
		case "tcp", "tcp4", "tcp6":
			return &TCPAddr{IP: ip.IP, Port: portnum, Zone: ip.Zone}
		case "udp", "udp4", "udp6":
			return &UDPAddr{IP: ip.IP, Port: portnum, Zone: ip.Zone}
		case "ip", "ip4", "ip6":
			return &IPAddr{IP: ip.IP, Zone: ip.Zone}
		default:
			panic("unexpected network: " + net)
		}
	}
	if host == "" {
		return addrList{inetaddr(IPAddr{})}, nil
	}
	// Try as a literal IP address.
	var ip IP
	if ip = parseIPv4(host); ip != nil {
		return addrList{inetaddr(IPAddr{IP: ip})}, nil
	}
	var zone string
	if ip, zone = parseIPv6(host, true); ip != nil {
		return addrList{inetaddr(IPAddr{IP: ip, Zone: zone})}, nil
	}
	// Try as a DNS name.
	ips, err := lookupIPDeadline(host, deadline)
	if err != nil {
		return nil, err
	}
	var filter func(IPAddr) bool
	if net != "" && net[len(net)-1] == '4' {
		filter = ipv4only
	}
	if net != "" && net[len(net)-1] == '6' {
		filter = ipv6only
	}
	return filterAddrList(filter, ips, inetaddr)
}

func zoneToString(zone int) string {
	if zone == 0 {
		return ""
	}
	if ifi, err := InterfaceByIndex(zone); err == nil {
		return ifi.Name
	}
	return uitoa(uint(zone))
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
