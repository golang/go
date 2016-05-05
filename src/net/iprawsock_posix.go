// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package net

import (
	"context"
	"syscall"
)

// BUG(mikio): On every POSIX platform, reads from the "ip4" network
// using the ReadFrom or ReadFromIP method might not return a complete
// IPv4 packet, including its header, even if there is space
// available. This can occur even in cases where Read or ReadMsgIP
// could return a complete packet. For this reason, it is recommended
// that you do not uses these methods if it is important to receive a
// full packet.
//
// The Go 1 compatibility guidelines make it impossible for us to
// change the behavior of these methods; use Read or ReadMsgIP
// instead.

func sockaddrToIP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &IPAddr{IP: sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		return &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	return nil
}

func (a *IPAddr) family() int {
	if a == nil || len(a.IP) <= IPv4len {
		return syscall.AF_INET
	}
	if a.IP.To4() != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

func (a *IPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return ipToSockaddr(family, a.IP, 0, a.Zone)
}

func (c *IPConn) readFrom(b []byte) (int, *IPAddr, error) {
	// TODO(cw,rsc): consider using readv if we know the family
	// type to avoid the header trim/copy
	var addr *IPAddr
	n, sa, err := c.fd.readFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
		n = stripIPv4Header(n, b)
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	return n, addr, err
}

func stripIPv4Header(n int, b []byte) int {
	if len(b) < 20 {
		return n
	}
	l := int(b[0]&0x0f) << 2
	if 20 > l || l > len(b) {
		return n
	}
	if b[0]>>4 != 4 {
		return n
	}
	copy(b, b[l:])
	return n - l
}

func (c *IPConn) readMsg(b, oob []byte) (n, oobn, flags int, addr *IPAddr, err error) {
	var sa syscall.Sockaddr
	n, oobn, flags, sa, err = c.fd.readMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	return
}

func (c *IPConn) writeTo(b []byte, addr *IPAddr) (int, error) {
	if c.fd.isConnected {
		return 0, ErrWriteToConnected
	}
	if addr == nil {
		return 0, errMissingAddress
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, err
	}
	return c.fd.writeTo(b, sa)
}

func (c *IPConn) writeMsg(b, oob []byte, addr *IPAddr) (n, oobn int, err error) {
	if c.fd.isConnected {
		return 0, 0, ErrWriteToConnected
	}
	if addr == nil {
		return 0, 0, errMissingAddress
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, 0, err
	}
	return c.fd.writeMsg(b, oob, sa)
}

func dialIP(ctx context.Context, netProto string, laddr, raddr *IPAddr) (*IPConn, error) {
	network, proto, err := parseNetwork(ctx, netProto)
	if err != nil {
		return nil, err
	}
	switch network {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(netProto)
	}
	if raddr == nil {
		return nil, errMissingAddress
	}
	fd, err := internetSocket(ctx, network, laddr, raddr, syscall.SOCK_RAW, proto, "dial")
	if err != nil {
		return nil, err
	}
	return newIPConn(fd), nil
}

func listenIP(ctx context.Context, netProto string, laddr *IPAddr) (*IPConn, error) {
	network, proto, err := parseNetwork(ctx, netProto)
	if err != nil {
		return nil, err
	}
	switch network {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(netProto)
	}
	fd, err := internetSocket(ctx, network, laddr, nil, syscall.SOCK_RAW, proto, "listen")
	if err != nil {
		return nil, err
	}
	return newIPConn(fd), nil
}
