// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package net

import "syscall"

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
		la TCPAddr
		ok bool
	}{
		// IPv6 communication capability
		{TCPAddr{IP: ParseIP("::1")}, false},
		// IPv6 IPv4-mapped address communication capability
		{TCPAddr{IP: IPv4(127, 0, 0, 1)}, false},
	}

	for i := range probes {
		s, err := syscall.Socket(syscall.AF_INET6, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
		if err != nil {
			continue
		}
		defer closesocket(s)
		sa, err := probes[i].la.toAddr().sockaddr(syscall.AF_INET6)
		if err != nil {
			continue
		}
		err = syscall.Bind(s, sa)
		if err != nil {
			continue
		}
		probes[i].ok = true
	}

	return probes[0].ok, probes[1].ok
}

// favoriteAddrFamily returns the appropriate address family to
// the given net, raddr, laddr and mode.  At first it figures
// address family out from the net.  If mode indicates "listen"
// and laddr.(type).IP is nil, it assumes that the user wants to
// make a passive connection with wildcard address family, both
// INET and INET6, and wildcard address.  Otherwise guess: if the
// addresses are IPv4 then returns INET, or else returns INET6.
func favoriteAddrFamily(net string, raddr, laddr sockaddr, mode string) int {
	switch net[len(net)-1] {
	case '4':
		return syscall.AF_INET
	case '6':
		return syscall.AF_INET6
	}

	if mode == "listen" {
		switch a := laddr.(type) {
		case *TCPAddr:
			if a.IP == nil && supportsIPv6 {
				return syscall.AF_INET6
			}
		case *UDPAddr:
			if a.IP == nil && supportsIPv6 {
				return syscall.AF_INET6
			}
		case *IPAddr:
			if a.IP == nil && supportsIPv6 {
				return syscall.AF_INET6
			}
		}
	}

	if (laddr == nil || laddr.family() == syscall.AF_INET) &&
		(raddr == nil || raddr.family() == syscall.AF_INET) {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

// Internet sockets (TCP, UDP)

// A sockaddr represents a TCP or UDP network address that can
// be converted into a syscall.Sockaddr.
type sockaddr interface {
	Addr
	sockaddr(family int) (syscall.Sockaddr, error)
	family() int
}

func internetSocket(net string, laddr, raddr sockaddr, socktype, proto int, mode string, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	var oserr error
	var la, ra syscall.Sockaddr
	family := favoriteAddrFamily(net, raddr, laddr, mode)
	if laddr != nil {
		if la, oserr = laddr.sockaddr(family); oserr != nil {
			goto Error
		}
	}
	if raddr != nil {
		if ra, oserr = raddr.sockaddr(family); oserr != nil {
			goto Error
		}
	}
	fd, oserr = socket(net, family, socktype, proto, la, ra, toAddr)
	if oserr != nil {
		goto Error
	}
	return fd, nil

Error:
	addr := raddr
	if mode == "listen" {
		addr = laddr
	}
	return nil, &OpError{mode, net, addr, oserr}
}

func ipToSockaddr(family int, ip IP, port int) (syscall.Sockaddr, error) {
	switch family {
	case syscall.AF_INET:
		if len(ip) == 0 {
			ip = IPv4zero
		}
		if ip = ip.To4(); ip == nil {
			return nil, InvalidAddrError("non-IPv4 address")
		}
		s := new(syscall.SockaddrInet4)
		for i := 0; i < IPv4len; i++ {
			s.Addr[i] = ip[i]
		}
		s.Port = port
		return s, nil
	case syscall.AF_INET6:
		if len(ip) == 0 {
			ip = IPv6zero
		}
		// IPv4 callers use 0.0.0.0 to mean "announce on any available address".
		// In IPv6 mode, Linux treats that as meaning "announce on 0.0.0.0",
		// which it refuses to do.  Rewrite to the IPv6 all zeros.
		if ip.Equal(IPv4zero) {
			ip = IPv6zero
		}
		if ip = ip.To16(); ip == nil {
			return nil, InvalidAddrError("non-IPv6 address")
		}
		s := new(syscall.SockaddrInet6)
		for i := 0; i < IPv6len; i++ {
			s.Addr[i] = ip[i]
		}
		s.Port = port
		return s, nil
	}
	return nil, InvalidAddrError("unexpected socket family")
}
