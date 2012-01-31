// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// UDP sockets

package net

import (
	"errors"
	"os"
	"syscall"
	"time"
)

var ErrWriteToConnected = errors.New("use of WriteTo with pre-connected UDP")

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
	fd *netFD
}

func newUDPConn(fd *netFD) *UDPConn { return &UDPConn{fd} }

func (c *UDPConn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *UDPConn) Read(b []byte) (int, error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Read(b)
}

// Write implements the Conn Write method.
func (c *UDPConn) Write(b []byte) (int, error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the UDP connection.
func (c *UDPConn) Close() error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// LocalAddr returns the local network address.
func (c *UDPConn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address, a *UDPAddr.
func (c *UDPConn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetDeadline implements the Conn SetDeadline method.
func (c *UDPConn) SetDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setDeadline(c.fd, t)
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *UDPConn) SetReadDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadDeadline(c.fd, t)
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *UDPConn) SetWriteDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteDeadline(c.fd, t)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *UDPConn) SetReadBuffer(bytes int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *UDPConn) SetWriteBuffer(bytes int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// UDP-specific methods.

// ReadFromUDP reads a UDP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUDP can be made to time out and return an error with Timeout() == true
// after a fixed time limit; see SetDeadline and SetReadDeadline.
func (c *UDPConn) ReadFromUDP(b []byte) (n int, addr *UDPAddr, err error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
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
		return 0, nil, os.EINVAL
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
		return 0, os.EINVAL
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
		return 0, os.EINVAL
	}
	a, ok := addr.(*UDPAddr)
	if !ok {
		return 0, &OpError{"write", c.fd.net, addr, os.EINVAL}
	}
	return c.WriteToUDP(b, a)
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *UDPConn) File() (f *os.File, err error) { return c.fd.dup() }

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
		return nil, &OpError{"listenmulticastudp", "udp", nil, errMissingAddress}
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

func leaveIPv4GroupUDP(c *UDPConn, ifi *Interface, ip IP) error {
	err := leaveIPv4Group(c.fd, ifi, ip)
	if err != nil {
		return &OpError{"leaveipv4group", c.fd.net, &IPAddr{ip}, err}
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

func leaveIPv6GroupUDP(c *UDPConn, ifi *Interface, ip IP) error {
	err := leaveIPv6Group(c.fd, ifi, ip)
	if err != nil {
		return &OpError{"leaveipv6group", c.fd.net, &IPAddr{ip}, err}
	}
	return nil
}
