// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package net

import (
	"syscall"
	"time"
)

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

func (a *IPAddr) isWildcard() bool {
	if a == nil || a.IP == nil {
		return true
	}
	return a.IP.IsUnspecified()
}

func (a *IPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	return ipToSockaddr(family, a.IP, 0, a.Zone)
}

func (a *IPAddr) toAddr() sockaddr {
	if a == nil { // nil *IPAddr
		return nil // nil interface
	}
	return a
}

// IPConn is the implementation of the Conn and PacketConn interfaces
// for IP network connections.
type IPConn struct {
	conn
}

func newIPConn(fd *netFD) *IPConn { return &IPConn{conn{fd}} }

// ReadFromIP reads an IP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *IPConn) ReadFromIP(b []byte) (int, *IPAddr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	// TODO(cw,rsc): consider using readv if we know the family
	// type to avoid the header trim/copy
	var addr *IPAddr
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
		if len(b) >= IPv4len { // discard ipv4 header
			hsize := (int(b[0]) & 0xf) * 4
			copy(b, b[hsize:])
			n -= hsize
		}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	return n, addr, err
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.ReadFromIP(b)
	return n, addr.toAddr(), err
}

// ReadMsgIP reads a packet from c, copying the payload into b and the
// associated out-of-band data into oob.  It returns the number of
// bytes copied into b, the number of bytes copied into oob, the flags
// that were set on the packet and the source address of the packet.
func (c *IPConn) ReadMsgIP(b, oob []byte) (n, oobn, flags int, addr *IPAddr, err error) {
	if !c.ok() {
		return 0, 0, 0, nil, syscall.EINVAL
	}
	var sa syscall.Sockaddr
	n, oobn, flags, sa, err = c.fd.ReadMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	return
}

// WriteToIP writes an IP packet to addr via c, copying the payload
// from b.
//
// WriteToIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, &OpError{"write", c.fd.net, addr, err}
	}
	return c.fd.WriteTo(b, sa)
}

// WriteTo implements the PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*IPAddr)
	if !ok {
		return 0, &OpError{"write", c.fd.net, addr, syscall.EINVAL}
	}
	return c.WriteToIP(b, a)
}

// WriteMsgIP writes a packet to addr via c, copying the payload from
// b and the associated out-of-band data from oob.  It returns the
// number of payload and out-of-band bytes written.
func (c *IPConn) WriteMsgIP(b, oob []byte, addr *IPAddr) (n, oobn int, err error) {
	if !c.ok() {
		return 0, 0, syscall.EINVAL
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, 0, &OpError{"write", c.fd.net, addr, err}
	}
	return c.fd.WriteMsg(b, oob, sa)
}

// DialIP connects to the remote address raddr on the network protocol
// netProto, which must be "ip", "ip4", or "ip6" followed by a colon
// and a protocol number or name.
func DialIP(netProto string, laddr, raddr *IPAddr) (*IPConn, error) {
	return dialIP(netProto, laddr, raddr, noDeadline)
}

func dialIP(netProto string, laddr, raddr *IPAddr, deadline time.Time) (*IPConn, error) {
	net, proto, err := parseNetwork(netProto)
	if err != nil {
		return nil, err
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(netProto)
	}
	if raddr == nil {
		return nil, &OpError{"dial", netProto, nil, errMissingAddress}
	}
	fd, err := internetSocket(net, laddr.toAddr(), raddr.toAddr(), deadline, syscall.SOCK_RAW, proto, "dial", sockaddrToIP)
	if err != nil {
		return nil, err
	}
	return newIPConn(fd), nil
}

// ListenIP listens for incoming IP packets addressed to the local
// address laddr.  The returned connection's ReadFrom and WriteTo
// methods can be used to receive and send IP packets with per-packet
// addressing.
func ListenIP(netProto string, laddr *IPAddr) (*IPConn, error) {
	net, proto, err := parseNetwork(netProto)
	if err != nil {
		return nil, err
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(netProto)
	}
	fd, err := internetSocket(net, laddr.toAddr(), nil, noDeadline, syscall.SOCK_RAW, proto, "listen", sockaddrToIP)
	if err != nil {
		return nil, err
	}
	return newIPConn(fd), nil
}
