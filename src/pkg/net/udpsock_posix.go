// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// UDP sockets

package net

import "syscall"

func sockaddrToUDP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &UDPAddr{sa.Addr[0:], sa.Port}
	case *syscall.SockaddrInet6:
		return &UDPAddr{sa.Addr[0:], sa.Port}
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

func (a *UDPAddr) isWildcard() bool {
	if a == nil || a.IP == nil {
		return true
	}
	return a.IP.IsUnspecified()
}

func (a *UDPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	return ipToSockaddr(family, a.IP, a.Port)
}

func (a *UDPAddr) toAddr() sockaddr {
	if a == nil { // nil *UDPAddr
		return nil // nil interface
	}
	return a
}

// UDPConn is the implementation of the Conn and PacketConn
// interfaces for UDP network connections.
type UDPConn struct {
	conn
}

func newUDPConn(fd *netFD) *UDPConn { return &UDPConn{conn{fd}} }

// UDP-specific methods.

// ReadFromUDP reads a UDP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUDP can be made to time out and return an error with Timeout() == true
// after a fixed time limit; see SetDeadline and SetReadDeadline.
func (c *UDPConn) ReadFromUDP(b []byte) (n int, addr *UDPAddr, err error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &UDPAddr{sa.Addr[0:], sa.Port}
	case *syscall.SockaddrInet6:
		addr = &UDPAddr{sa.Addr[0:], sa.Port}
	}
	return
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UDPConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, uaddr, err := c.ReadFromUDP(b)
	return n, uaddr.toAddr(), err
}

// WriteToUDP writes a UDP packet to addr via c, copying the payload from b.
//
// WriteToUDP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetDeadline and SetWriteDeadline.
// On packet-oriented connections, write timeouts are rare.
func (c *UDPConn) WriteToUDP(b []byte, addr *UDPAddr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	if c.fd.isConnected {
		return 0, &OpError{"write", c.fd.net, addr, ErrWriteToConnected}
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, &OpError{"write", c.fd.net, addr, err}
	}
	return c.fd.WriteTo(b, sa)
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UDPConn) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*UDPAddr)
	if !ok {
		return 0, &OpError{"write", c.fd.net, addr, syscall.EINVAL}
	}
	return c.WriteToUDP(b, a)
}

// DialUDP connects to the remote address raddr on the network net,
// which must be "udp", "udp4", or "udp6".  If laddr is not nil, it is used
// as the local address for the connection.
func DialUDP(net string, laddr, raddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}
	fd, err := internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_DGRAM, 0, "dial", sockaddrToUDP)
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}

// ListenUDP listens for incoming UDP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send UDP
// packets with per-packet addressing.
func ListenUDP(net string, laddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		return nil, &OpError{"listen", net, nil, errMissingAddress}
	}
	fd, err := internetSocket(net, laddr.toAddr(), nil, syscall.SOCK_DGRAM, 0, "listen", sockaddrToUDP)
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}

// ListenMulticastUDP listens for incoming multicast UDP packets
// addressed to the group address gaddr on ifi, which specifies
// the interface to join.  ListenMulticastUDP uses default
// multicast interface if ifi is nil.
func ListenMulticastUDP(net string, ifi *Interface, gaddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if gaddr == nil || gaddr.IP == nil {
		return nil, &OpError{"listenmulticast", net, nil, errMissingAddress}
	}
	fd, err := internetSocket(net, gaddr.toAddr(), nil, syscall.SOCK_DGRAM, 0, "listen", sockaddrToUDP)
	if err != nil {
		return nil, err
	}
	c := newUDPConn(fd)
	ip4 := gaddr.IP.To4()
	if ip4 != nil {
		err := listenIPv4MulticastUDP(c, ifi, ip4)
		if err != nil {
			c.Close()
			return nil, err
		}
	} else {
		err := listenIPv6MulticastUDP(c, ifi, gaddr.IP)
		if err != nil {
			c.Close()
			return nil, err
		}
	}
	return c, nil
}

func listenIPv4MulticastUDP(c *UDPConn, ifi *Interface, ip IP) error {
	if ifi != nil {
		err := setIPv4MulticastInterface(c.fd, ifi)
		if err != nil {
			return err
		}
	}
	err := setIPv4MulticastLoopback(c.fd, false)
	if err != nil {
		return err
	}
	err = joinIPv4GroupUDP(c, ifi, ip)
	if err != nil {
		return err
	}
	return nil
}

func listenIPv6MulticastUDP(c *UDPConn, ifi *Interface, ip IP) error {
	if ifi != nil {
		err := setIPv6MulticastInterface(c.fd, ifi)
		if err != nil {
			return err
		}
	}
	err := setIPv6MulticastLoopback(c.fd, false)
	if err != nil {
		return err
	}
	err = joinIPv6GroupUDP(c, ifi, ip)
	if err != nil {
		return err
	}
	return nil
}

func joinIPv4GroupUDP(c *UDPConn, ifi *Interface, ip IP) error {
	err := joinIPv4Group(c.fd, ifi, ip)
	if err != nil {
		return &OpError{"joinipv4group", c.fd.net, &IPAddr{ip}, err}
	}
	return nil
}

func joinIPv6GroupUDP(c *UDPConn, ifi *Interface, ip IP) error {
	err := joinIPv6Group(c.fd, ifi, ip)
	if err != nil {
		return &OpError{"joinipv6group", c.fd.net, &IPAddr{ip}, err}
	}
	return nil
}
