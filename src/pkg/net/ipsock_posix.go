// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

// Internet protocol family sockets for POSIX

package net

import (
	"syscall"
	"time"
)

func probeIPv4Stack() bool {
	s, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
	switch err {
	case syscall.EAFNOSUPPORT, syscall.EPROTONOSUPPORT:
		return false
	case nil:
		closesocket(s)
	}
	return true
}

// Should we try to use the IPv4 socket interface if we're
// only dealing with IPv4 sockets?  As long as the host system
// understands IPv6, it's okay to pass IPv4 addresses to the IPv6
// interface.  That simplifies our code and is most general.
// Unfortunately, we need to run on kernels built without IPv6
// support too.  So probe the kernel to figure it out.
//
// probeIPv6Stack probes both basic IPv6 capability and IPv6 IPv4-
// mapping capability which is controlled by IPV6_V6ONLY socket
// option and/or kernel state "net.inet6.ip6.v6only".
// It returns two boolean values.  If the first boolean value is
// true, kernel supports basic IPv6 functionality.  If the second
// boolean value is true, kernel supports IPv6 IPv4-mapping.
func probeIPv6Stack() (supportsIPv6, supportsIPv4map bool) {
	var probes = []struct {
		laddr TCPAddr
		value int
		ok    bool
	}{
		// IPv6 communication capability
		{laddr: TCPAddr{IP: ParseIP("::1")}, value: 1},
		// IPv6 IPv4-mapped address communication capability
		{laddr: TCPAddr{IP: IPv4(127, 0, 0, 1)}, value: 0},
	}

	for i := range probes {
		s, err := syscall.Socket(syscall.AF_INET6, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
		if err != nil {
			continue
		}
		defer closesocket(s)
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, probes[i].value)
		sa, err := probes[i].laddr.sockaddr(syscall.AF_INET6)
		if err != nil {
			continue
		}
		if err := syscall.Bind(s, sa); err != nil {
			continue
		}
		probes[i].ok = true
	}

	return probes[0].ok, probes[1].ok
}

// favoriteAddrFamily returns the appropriate address family to
// the given net, laddr, raddr and mode.  At first it figures
// address family out from the net.  If mode indicates "listen"
// and laddr is a wildcard, it assumes that the user wants to
// make a passive connection with a wildcard address family, both
// AF_INET and AF_INET6, and a wildcard address like following:
//
//	1. A wild-wild listen, "tcp" + ""
//	If the platform supports both IPv6 and IPv6 IPv4-mapping
//	capabilities, we assume that the user want to listen on
//	both IPv4 and IPv6 wildcard address over an AF_INET6
//	socket with IPV6_V6ONLY=0.  Otherwise we prefer an IPv4
//	wildcard address listen over an AF_INET socket.
//
//	2. A wild-ipv4wild listen, "tcp" + "0.0.0.0"
//	Same as 1.
//
//	3. A wild-ipv6wild listen, "tcp" + "[::]"
//	Almost same as 1 but we prefer an IPv6 wildcard address
//	listen over an AF_INET6 socket with IPV6_V6ONLY=0 when
//	the platform supports IPv6 capability but not IPv6 IPv4-
//	mapping capability.
//
//	4. A ipv4-ipv4wild listen, "tcp4" + "" or "0.0.0.0"
//	We use an IPv4 (AF_INET) wildcard address listen.
//
//	5. A ipv6-ipv6wild listen, "tcp6" + "" or "[::]"
//	We use an IPv6 (AF_INET6, IPV6_V6ONLY=1) wildcard address
//	listen.
//
// Otherwise guess: if the addresses are IPv4 then returns AF_INET,
// or else returns AF_INET6.  It also returns a boolean value what
// designates IPV6_V6ONLY option.
//
// Note that OpenBSD allows neither "net.inet6.ip6.v6only=1" change
// nor IPPROTO_IPV6 level IPV6_V6ONLY socket option setting.
func favoriteAddrFamily(net string, laddr, raddr sockaddr, mode string) (family int, ipv6only bool) {
	switch net[len(net)-1] {
	case '4':
		return syscall.AF_INET, false
	case '6':
		return syscall.AF_INET6, true
	}

	if mode == "listen" && (laddr == nil || laddr.isWildcard()) {
		if supportsIPv4map {
			return syscall.AF_INET6, false
		}
		if laddr == nil {
			return syscall.AF_INET, false
		}
		return laddr.family(), false
	}

	if (laddr == nil || laddr.family() == syscall.AF_INET) &&
		(raddr == nil || raddr.family() == syscall.AF_INET) {
		return syscall.AF_INET, false
	}
	return syscall.AF_INET6, false
}

// Internet sockets (TCP, UDP, IP)

func internetSocket(net string, laddr, raddr sockaddr, deadline time.Time, sotype, proto int, mode string) (fd *netFD, err error) {
	family, ipv6only := favoriteAddrFamily(net, laddr, raddr, mode)
	return socket(net, family, sotype, proto, ipv6only, laddr, raddr, deadline)
}

func ipToSockaddr(family int, ip IP, port int, zone string) (syscall.Sockaddr, error) {
	switch family {
	case syscall.AF_INET:
		if len(ip) == 0 {
			ip = IPv4zero
		}
		if ip = ip.To4(); ip == nil {
			return nil, InvalidAddrError("non-IPv4 address")
		}
		sa := new(syscall.SockaddrInet4)
		for i := 0; i < IPv4len; i++ {
			sa.Addr[i] = ip[i]
		}
		sa.Port = port
		return sa, nil
	case syscall.AF_INET6:
		if len(ip) == 0 {
			ip = IPv6zero
		}
		// IPv4 callers use 0.0.0.0 to mean "announce on any available address".
		// In IPv6 mode, Linux treats that as meaning "announce on 0.0.0.0",
		// which it refuses to do.  Rewrite to the IPv6 unspecified address.
		if ip.Equal(IPv4zero) {
			ip = IPv6zero
		}
		if ip = ip.To16(); ip == nil {
			return nil, InvalidAddrError("non-IPv6 address")
		}
		sa := new(syscall.SockaddrInet6)
		for i := 0; i < IPv6len; i++ {
			sa.Addr[i] = ip[i]
		}
		sa.Port = port
		sa.ZoneId = uint32(zoneToInt(zone))
		return sa, nil
	}
	return nil, InvalidAddrError("unexpected socket family")
}
