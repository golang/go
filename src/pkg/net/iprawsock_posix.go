// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// (Raw) IP sockets

package net

import (
	"errors"
	"os"
	"syscall"
)

func sockaddrToIP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &IPAddr{sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		return &IPAddr{sa.Addr[0:]}
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
	return ipToSockaddr(family, a.IP, 0)
}

func (a *IPAddr) toAddr() sockaddr {
	if a == nil { // nil *IPAddr
		return nil // nil interface
	}
	return a
}

// IPConn is the implementation of the Conn and PacketConn
// interfaces for IP network connections.
type IPConn struct {
	fd *netFD
}

func newIPConn(fd *netFD) *IPConn { return &IPConn{fd} }

func (c *IPConn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the net.Conn Read method.
func (c *IPConn) Read(b []byte) (n int, err error) {
	n, _, err = c.ReadFrom(b)
	return
}

// Write implements the net.Conn Write method.
func (c *IPConn) Write(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the IP connection.
func (c *IPConn) Close() error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// LocalAddr returns the local network address.
func (c *IPConn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address, a *IPAddr.
func (c *IPConn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetTimeout implements the net.Conn SetTimeout method.
func (c *IPConn) SetTimeout(nsec int64) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setTimeout(c.fd, nsec)
}

// SetReadTimeout implements the net.Conn SetReadTimeout method.
func (c *IPConn) SetReadTimeout(nsec int64) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadTimeout(c.fd, nsec)
}

// SetWriteTimeout implements the net.Conn SetWriteTimeout method.
func (c *IPConn) SetWriteTimeout(nsec int64) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteTimeout(c.fd, nsec)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *IPConn) SetReadBuffer(bytes int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *IPConn) SetWriteBuffer(bytes int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// IP-specific methods.

// ReadFromIP reads a IP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetTimeout and
// SetReadTimeout.
func (c *IPConn) ReadFromIP(b []byte) (n int, addr *IPAddr, err error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	// TODO(cw,rsc): consider using readv if we know the family
	// type to avoid the header trim/copy
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{sa.Addr[0:]}
		if len(b) >= IPv4len { // discard ipv4 header
			hsize := (int(b[0]) & 0xf) * 4
			copy(b, b[hsize:])
			n -= hsize
		}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{sa.Addr[0:]}
	}
	return
}

// ReadFrom implements the net.PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (n int, addr Addr, err error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	n, uaddr, err := c.ReadFromIP(b)
	return n, uaddr.toAddr(), err
}

// WriteToIP writes a IP packet to addr via c, copying the payload from b.
//
// WriteToIP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetTimeout and SetWriteTimeout.
// On packet-oriented connections, write timeouts are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	sa, err1 := addr.sockaddr(c.fd.family)
	if err1 != nil {
		return 0, &OpError{Op: "write", Net: "ip", Addr: addr, Err: err1}
	}
	return c.fd.WriteTo(b, sa)
}

// WriteTo implements the net.PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	a, ok := addr.(*IPAddr)
	if !ok {
		return 0, &OpError{"writeto", "ip", addr, os.EINVAL}
	}
	return c.WriteToIP(b, a)
}

func splitNetProto(netProto string) (net string, proto int, err error) {
	i := last(netProto, ':')
	if i < 0 { // no colon
		return "", 0, errors.New("no IP protocol specified")
	}
	net = netProto[0:i]
	protostr := netProto[i+1:]
	proto, i, ok := dtoi(protostr, 0)
	if !ok || i != len(protostr) {
		proto, err = lookupProtocol(protostr)
		if err != nil {
			return "", 0, err
		}
	}
	return net, proto, nil
}

// DialIP connects to the remote address raddr on the network protocol netProto,
// which must be "ip", "ip4", or "ip6" followed by a colon and a protocol number or name.
func DialIP(netProto string, laddr, raddr *IPAddr) (c *IPConn, err error) {
	net, proto, err := splitNetProto(netProto)
	if err != nil {
		return
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", "ip", nil, errMissingAddress}
	}
	fd, e := internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_RAW, proto, "dial", sockaddrToIP)
	if e != nil {
		return nil, e
	}
	return newIPConn(fd), nil
}

// ListenIP listens for incoming IP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send IP
// packets with per-packet addressing.
func ListenIP(netProto string, laddr *IPAddr) (c *IPConn, err error) {
	net, proto, err := splitNetProto(netProto)
	if err != nil {
		return
	}
	switch net {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(net)
	}
	fd, e := internetSocket(net, laddr.toAddr(), nil, syscall.SOCK_RAW, proto, "listen", sockaddrToIP)
	if e != nil {
		return nil, e
	}
	return newIPConn(fd), nil
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *IPConn) File() (f *os.File, err error) { return c.fd.dup() }
