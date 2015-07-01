// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package net

import (
	"syscall"
	"time"
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
	n, sa, err := c.fd.readFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
		n = stripIPv4Header(n, b)
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	if err != nil {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
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

// ReadFrom implements the PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.ReadFromIP(b)
	if addr == nil {
		return n, nil, err
	}
	return n, addr, err
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
	n, oobn, flags, sa, err = c.fd.readMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{IP: sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{IP: sa.Addr[0:], Zone: zoneToString(int(sa.ZoneId))}
	}
	if err != nil {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
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
	if c.fd.isConnected {
		return 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: ErrWriteToConnected}
	}
	if addr == nil {
		return 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: nil, Err: errMissingAddress}
	}
	sa, err := addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	n, err := c.fd.writeTo(b, sa)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	return n, err
}

// WriteTo implements the PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*IPAddr)
	if !ok {
		return 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr, Err: syscall.EINVAL}
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
	if c.fd.isConnected {
		return 0, 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: ErrWriteToConnected}
	}
	if addr == nil {
		return 0, 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: nil, Err: errMissingAddress}
	}
	var sa syscall.Sockaddr
	sa, err = addr.sockaddr(c.fd.family)
	if err != nil {
		return 0, 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	n, oobn, err = c.fd.writeMsg(b, oob, sa)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	return
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
		return nil, &OpError{Op: "dial", Net: netProto, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: err}
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, &OpError{Op: "dial", Net: netProto, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: UnknownNetworkError(netProto)}
	}
	if raddr == nil {
		return nil, &OpError{Op: "dial", Net: netProto, Source: laddr.opAddr(), Addr: nil, Err: errMissingAddress}
	}
	fd, err := internetSocket(net, laddr, raddr, deadline, syscall.SOCK_RAW, proto, "dial")
	if err != nil {
		return nil, &OpError{Op: "dial", Net: netProto, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: err}
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
		return nil, &OpError{Op: "listen", Net: netProto, Source: nil, Addr: laddr.opAddr(), Err: err}
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, &OpError{Op: "listen", Net: netProto, Source: nil, Addr: laddr.opAddr(), Err: UnknownNetworkError(netProto)}
	}
	fd, err := internetSocket(net, laddr, nil, noDeadline, syscall.SOCK_RAW, proto, "listen")
	if err != nil {
		return nil, &OpError{Op: "listen", Net: netProto, Source: nil, Addr: laddr.opAddr(), Err: err}
	}
	return newIPConn(fd), nil
}
