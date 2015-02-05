// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package net

import (
	"syscall"
	"time"
)

func sockaddrToUDP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &UDPAddr{IP: sa.Addr[0:], Port: sa.Port}
	case *syscall.SockaddrInet6:
		return &UDPAddr{IP: sa.Addr[0:], Port: sa.Port, Zone: zoneToString(int(sa.ZoneId))}
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
	if a == nil {
		return nil, nil
	}
	return ipToSockaddr(family, a.IP, a.Port, a.Zone)
}

// UDPConn is the implementation of the Conn and PacketConn interfaces
// for UDP network connections.
type UDPConn struct {
	conn
}

func newUDPConn(fd *netFD) *UDPConn { return &UDPConn{conn{fd}} }

// ReadFromUDP reads a UDP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUDP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *UDPConn) ReadFromUDP(b []byte) (n int, addr *UDPAddr, err error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, sa, err := c.fd.readFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &UDPAddr{IP: sa.Addr[0:], Port: sa.Port}
	case *syscall.SockaddrInet6:
		addr = &UDPAddr{IP: sa.Addr[0:], Port: sa.Port, Zone: zoneToString(int(sa.ZoneId))}
	}
	return
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UDPConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.ReadFromUDP(b)
	return n, addr.toAddr(), err
}

// ReadMsgUDP reads a packet from c, copying the payload into b and
// the associated out-of-band data into oob.  It returns the number
// of bytes copied into b, the number of bytes copied into oob, the
// flags that were set on the packet and the source address of the
// packet.
func (c *UDPConn) ReadMsgUDP(b, oob []byte) (n, oobn, flags int, addr *UDPAddr, err error) {
	if !c.ok() {
		return 0, 0, 0, nil, syscall.EINVAL
	}
	var sa syscall.Sockaddr
	n, oobn, flags, sa, err = c.fd.readMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &UDPAddr{IP: sa.Addr[0:], Port: sa.Port}
	case *syscall.SockaddrInet6:
		addr = &UDPAddr{IP: sa.Addr[0:], Port: sa.Port, Zone: zoneToString(int(sa.ZoneId))}
	}
	return
}

// WriteToUDP writes a UDP packet to addr via c, copying the payload
// from b.
//
// WriteToUDP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *UDPConn) WriteToUDP(b []byte, addr *UDPAddr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	if c.fd.isConnected {
		return 0, &OpError{"write", c.fd.net, addr, ErrWriteToConnected}
	}
	if addr == nil {
		return 0, &OpError{Op: "write", Net: c.fd.net, Addr: nil, Err: errMissingAddress}
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, &OpError{"write", c.fd.net, addr, err}
	}
	return c.fd.writeTo(b, sa)
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

// WriteMsgUDP writes a packet to addr via c if c isn't connected, or
// to c's remote destination address if c is connected (in which case
// addr must be nil).  The payload is copied from b and the associated
// out-of-band data is copied from oob.  It returns the number of
// payload and out-of-band bytes written.
func (c *UDPConn) WriteMsgUDP(b, oob []byte, addr *UDPAddr) (n, oobn int, err error) {
	if !c.ok() {
		return 0, 0, syscall.EINVAL
	}
	if c.fd.isConnected && addr != nil {
		return 0, 0, &OpError{"write", c.fd.net, addr, ErrWriteToConnected}
	}
	if !c.fd.isConnected && addr == nil {
		return 0, 0, &OpError{Op: "write", Net: c.fd.net, Addr: nil, Err: errMissingAddress}
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, 0, &OpError{"write", c.fd.net, addr, err}
	}
	return c.fd.writeMsg(b, oob, sa)
}

// DialUDP connects to the remote address raddr on the network net,
// which must be "udp", "udp4", or "udp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialUDP(net string, laddr, raddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: UnknownNetworkError(net)}
	}
	if raddr == nil {
		return nil, &OpError{Op: "dial", Net: net, Addr: nil, Err: errMissingAddress}
	}
	return dialUDP(net, laddr, raddr, noDeadline)
}

func dialUDP(net string, laddr, raddr *UDPAddr, deadline time.Time) (*UDPConn, error) {
	fd, err := internetSocket(net, laddr, raddr, deadline, syscall.SOCK_DGRAM, 0, "dial")
	if err != nil {
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: err}
	}
	return newUDPConn(fd), nil
}

// ListenUDP listens for incoming UDP packets addressed to the local
// address laddr.  Net must be "udp", "udp4", or "udp6".  If laddr has
// a port of 0, ListenUDP will choose an available port.
// The LocalAddr method of the returned UDPConn can be used to
// discover the port.  The returned connection's ReadFrom and WriteTo
// methods can be used to receive and send UDP packets with per-packet
// addressing.
func ListenUDP(net string, laddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: UnknownNetworkError(net)}
	}
	if laddr == nil {
		laddr = &UDPAddr{}
	}
	fd, err := internetSocket(net, laddr, nil, noDeadline, syscall.SOCK_DGRAM, 0, "listen")
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: err}
	}
	return newUDPConn(fd), nil
}

// ListenMulticastUDP listens for incoming multicast UDP packets
// addressed to the group address gaddr on ifi, which specifies the
// interface to join.  ListenMulticastUDP uses default multicast
// interface if ifi is nil.
func ListenMulticastUDP(net string, ifi *Interface, gaddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: gaddr, Err: UnknownNetworkError(net)}
	}
	if gaddr == nil || gaddr.IP == nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: nil, Err: errMissingAddress}
	}
	fd, err := internetSocket(net, gaddr, nil, noDeadline, syscall.SOCK_DGRAM, 0, "listen")
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: gaddr, Err: err}
	}
	c := newUDPConn(fd)
	if ip4 := gaddr.IP.To4(); ip4 != nil {
		if err := listenIPv4MulticastUDP(c, ifi, ip4); err != nil {
			c.Close()
			return nil, &OpError{Op: "listen", Net: net, Addr: &IPAddr{IP: ip4}, Err: err}
		}
	} else {
		if err := listenIPv6MulticastUDP(c, ifi, gaddr.IP); err != nil {
			c.Close()
			return nil, &OpError{Op: "listen", Net: net, Addr: &IPAddr{IP: gaddr.IP}, Err: err}
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
