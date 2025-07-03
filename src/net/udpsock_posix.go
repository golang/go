// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || js || wasip1 || windows

package net

import (
	"context"
	"net/netip"
	"syscall"
)

func sockaddrToUDP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &UDPAddr{IP: sa.Addr[0:], Port: sa.Port}
	case *syscall.SockaddrInet6:
		return &UDPAddr{IP: sa.Addr[0:], Port: sa.Port, Zone: zoneCache.name(int(sa.ZoneId))}
	}
	return nil
}

func (a *UDPAddr) family() int {
	if a == nil || len(a.IP) <= IPv4len {
		return syscall.AF_INET
	}
	if a.IP.To4() != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

func (a *UDPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return ipToSockaddr(family, a.IP, a.Port, a.Zone)
}

func (a *UDPAddr) toLocal(net string) sockaddr {
	return &UDPAddr{loopbackIP(net), a.Port, a.Zone}
}

func (c *UDPConn) readFrom(b []byte, addr *UDPAddr) (int, *UDPAddr, error) {
	var n int
	var err error
	switch c.fd.family {
	case syscall.AF_INET:
		var from syscall.SockaddrInet4
		n, err = c.fd.readFromInet4(b, &from)
		if err == nil {
			ip := from.Addr // copy from.Addr; ip escapes, so this line allocates 4 bytes
			*addr = UDPAddr{IP: ip[:], Port: from.Port}
		}
	case syscall.AF_INET6:
		var from syscall.SockaddrInet6
		n, err = c.fd.readFromInet6(b, &from)
		if err == nil {
			ip := from.Addr // copy from.Addr; ip escapes, so this line allocates 16 bytes
			*addr = UDPAddr{IP: ip[:], Port: from.Port, Zone: zoneCache.name(int(from.ZoneId))}
		}
	}
	if err != nil {
		// No sockaddr, so don't return UDPAddr.
		addr = nil
	}
	return n, addr, err
}

func (c *UDPConn) readFromAddrPort(b []byte) (n int, addr netip.AddrPort, err error) {
	var ip netip.Addr
	var port int
	switch c.fd.family {
	case syscall.AF_INET:
		var from syscall.SockaddrInet4
		n, err = c.fd.readFromInet4(b, &from)
		if err == nil {
			ip = netip.AddrFrom4(from.Addr)
			port = from.Port
		}
	case syscall.AF_INET6:
		var from syscall.SockaddrInet6
		n, err = c.fd.readFromInet6(b, &from)
		if err == nil {
			ip = netip.AddrFrom16(from.Addr).WithZone(zoneCache.name(int(from.ZoneId)))
			port = from.Port
		}
	}
	if err == nil {
		addr = netip.AddrPortFrom(ip, uint16(port))
	}
	return n, addr, err
}

func (c *UDPConn) readMsg(b, oob []byte) (n, oobn, flags int, addr netip.AddrPort, err error) {
	switch c.fd.family {
	case syscall.AF_INET:
		var sa syscall.SockaddrInet4
		n, oobn, flags, err = c.fd.readMsgInet4(b, oob, 0, &sa)
		ip := netip.AddrFrom4(sa.Addr)
		addr = netip.AddrPortFrom(ip, uint16(sa.Port))
	case syscall.AF_INET6:
		var sa syscall.SockaddrInet6
		n, oobn, flags, err = c.fd.readMsgInet6(b, oob, 0, &sa)
		ip := netip.AddrFrom16(sa.Addr).WithZone(zoneCache.name(int(sa.ZoneId)))
		addr = netip.AddrPortFrom(ip, uint16(sa.Port))
	}
	return
}

func (c *UDPConn) writeTo(b []byte, addr *UDPAddr) (int, error) {
	if c.fd.isConnected {
		return 0, ErrWriteToConnected
	}
	if addr == nil {
		return 0, errMissingAddress
	}

	switch c.fd.family {
	case syscall.AF_INET:
		sa, err := ipToSockaddrInet4(addr.IP, addr.Port)
		if err != nil {
			return 0, err
		}
		return c.fd.writeToInet4(b, &sa)
	case syscall.AF_INET6:
		sa, err := ipToSockaddrInet6(addr.IP, addr.Port, addr.Zone)
		if err != nil {
			return 0, err
		}
		return c.fd.writeToInet6(b, &sa)
	default:
		return 0, &AddrError{Err: "invalid address family", Addr: addr.IP.String()}
	}
}

func (c *UDPConn) writeToAddrPort(b []byte, addr netip.AddrPort) (int, error) {
	if c.fd.isConnected {
		return 0, ErrWriteToConnected
	}
	if !addr.IsValid() {
		return 0, errMissingAddress
	}

	switch c.fd.family {
	case syscall.AF_INET:
		sa, err := addrPortToSockaddrInet4(addr)
		if err != nil {
			return 0, err
		}
		return c.fd.writeToInet4(b, &sa)
	case syscall.AF_INET6:
		sa, err := addrPortToSockaddrInet6(addr)
		if err != nil {
			return 0, err
		}
		return c.fd.writeToInet6(b, &sa)
	default:
		return 0, &AddrError{Err: "invalid address family", Addr: addr.Addr().String()}
	}
}

func (c *UDPConn) writeMsg(b, oob []byte, addr *UDPAddr) (n, oobn int, err error) {
	if c.fd.isConnected && addr != nil {
		return 0, 0, ErrWriteToConnected
	}
	if !c.fd.isConnected && addr == nil {
		return 0, 0, errMissingAddress
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, 0, err
	}
	return c.fd.writeMsg(b, oob, sa)
}

func (c *UDPConn) writeMsgAddrPort(b, oob []byte, addr netip.AddrPort) (n, oobn int, err error) {
	if c.fd.isConnected && addr.IsValid() {
		return 0, 0, ErrWriteToConnected
	}
	if !c.fd.isConnected && !addr.IsValid() {
		return 0, 0, errMissingAddress
	}

	switch c.fd.family {
	case syscall.AF_INET:
		sa, err := addrPortToSockaddrInet4(addr)
		if err != nil {
			return 0, 0, err
		}
		return c.fd.writeMsgInet4(b, oob, &sa)
	case syscall.AF_INET6:
		sa, err := addrPortToSockaddrInet6(addr)
		if err != nil {
			return 0, 0, err
		}
		return c.fd.writeMsgInet6(b, oob, &sa)
	default:
		return 0, 0, &AddrError{Err: "invalid address family", Addr: addr.Addr().String()}
	}
}

func (sd *sysDialer) dialUDP(ctx context.Context, laddr, raddr *UDPAddr) (*UDPConn, error) {
	ctrlCtxFn := sd.Dialer.ControlContext
	if ctrlCtxFn == nil && sd.Dialer.Control != nil {
		ctrlCtxFn = func(ctx context.Context, network, address string, c syscall.RawConn) error {
			return sd.Dialer.Control(network, address, c)
		}
	}
	fd, err := internetSocket(ctx, sd.network, laddr, raddr, syscall.SOCK_DGRAM, 0, "dial", ctrlCtxFn)
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}

func (sl *sysListener) listenUDP(ctx context.Context, laddr *UDPAddr) (*UDPConn, error) {
	var ctrlCtxFn func(ctx context.Context, network, address string, c syscall.RawConn) error
	if sl.ListenConfig.Control != nil {
		ctrlCtxFn = func(ctx context.Context, network, address string, c syscall.RawConn) error {
			return sl.ListenConfig.Control(network, address, c)
		}
	}
	fd, err := internetSocket(ctx, sl.network, laddr, nil, syscall.SOCK_DGRAM, 0, "listen", ctrlCtxFn)
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}

func (sl *sysListener) listenMulticastUDP(ctx context.Context, ifi *Interface, gaddr *UDPAddr) (*UDPConn, error) {
	var ctrlCtxFn func(ctx context.Context, network, address string, c syscall.RawConn) error
	if sl.ListenConfig.Control != nil {
		ctrlCtxFn = func(ctx context.Context, network, address string, c syscall.RawConn) error {
			return sl.ListenConfig.Control(network, address, c)
		}
	}
	fd, err := internetSocket(ctx, sl.network, gaddr, nil, syscall.SOCK_DGRAM, 0, "listen", ctrlCtxFn)
	if err != nil {
		return nil, err
	}
	c := newUDPConn(fd)
	if ip4 := gaddr.IP.To4(); ip4 != nil {
		if err := listenIPv4MulticastUDP(c, ifi, ip4); err != nil {
			c.Close()
			return nil, err
		}
	} else {
		if err := listenIPv6MulticastUDP(c, ifi, gaddr.IP); err != nil {
			c.Close()
			return nil, err
		}
	}
	return c, nil
}

func listenIPv4MulticastUDP(c *UDPConn, ifi *Interface, ip IP) error {
	if ifi != nil {
		if err := setIPv4MulticastInterface(c.fd, ifi); err != nil {
			return err
		}
	}
	if err := setIPv4MulticastLoopback(c.fd, false); err != nil {
		return err
	}
	if err := joinIPv4Group(c.fd, ifi, ip); err != nil {
		return err
	}
	return nil
}

func listenIPv6MulticastUDP(c *UDPConn, ifi *Interface, ip IP) error {
	if ifi != nil {
		if err := setIPv6MulticastInterface(c.fd, ifi); err != nil {
			return err
		}
	}
	if err := setIPv6MulticastLoopback(c.fd, false); err != nil {
		return err
	}
	if err := joinIPv6Group(c.fd, ifi, ip); err != nil {
		return err
	}
	return nil
}
