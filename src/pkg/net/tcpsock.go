// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP sockets

package net

import (
	"os"
	"syscall"
)

func sockaddrToTCP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &TCPAddr{sa.Addr[0:], sa.Port}
	case *syscall.SockaddrInet6:
		return &TCPAddr{sa.Addr[0:], sa.Port}
	}
	return nil
}

// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
	IP   IP
	Port int
}

// Network returns the address's network name, "tcp".
func (a *TCPAddr) Network() string { return "tcp" }

func (a *TCPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return joinHostPort(a.IP.String(), itoa(a.Port))
}

func (a *TCPAddr) family() int {
	if a == nil || len(a.IP) <= 4 {
		return syscall.AF_INET
	}
	if ip := a.IP.To4(); ip != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

func (a *TCPAddr) sockaddr(family int) (syscall.Sockaddr, os.Error) {
	return ipToSockaddr(family, a.IP, a.Port)
}

func (a *TCPAddr) toAddr() sockaddr {
	if a == nil { // nil *TCPAddr
		return nil // nil interface
	}
	return a
}

// ResolveTCPAddr parses addr as a TCP address of the form
// host:port and resolves domain names or port names to
// numeric addresses.  A literal IPv6 host address must be
// enclosed in square brackets, as in "[::]:80".
func ResolveTCPAddr(addr string) (*TCPAddr, os.Error) {
	ip, port, err := hostPortToIP("tcp", addr)
	if err != nil {
		return nil, err
	}
	return &TCPAddr{ip, port}, nil
}

// TCPConn is an implementation of the Conn interface
// for TCP network connections.
type TCPConn struct {
	fd *netFD
}

func newTCPConn(fd *netFD) *TCPConn {
	c := &TCPConn{fd}
	c.SetNoDelay(true)
	return c
}

func (c *TCPConn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the net.Conn Read method.
func (c *TCPConn) Read(b []byte) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Read(b)
}

// Write implements the net.Conn Write method.
func (c *TCPConn) Write(b []byte) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the TCP connection.
func (c *TCPConn) Close() os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// LocalAddr returns the local network address, a *TCPAddr.
func (c *TCPConn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address, a *TCPAddr.
func (c *TCPConn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetTimeout implements the net.Conn SetTimeout method.
func (c *TCPConn) SetTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setTimeout(c.fd, nsec)
}

// SetReadTimeout implements the net.Conn SetReadTimeout method.
func (c *TCPConn) SetReadTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadTimeout(c.fd, nsec)
}

// SetWriteTimeout implements the net.Conn SetWriteTimeout method.
func (c *TCPConn) SetWriteTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteTimeout(c.fd, nsec)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *TCPConn) SetReadBuffer(bytes int) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *TCPConn) SetWriteBuffer(bytes int) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// SetLinger sets the behavior of Close() on a connection
// which still has data waiting to be sent or to be acknowledged.
//
// If sec < 0 (the default), Close returns immediately and
// the operating system finishes sending the data in the background.
//
// If sec == 0, Close returns immediately and the operating system
// discards any unsent or unacknowledged data.
//
// If sec > 0, Close blocks for at most sec seconds waiting for
// data to be sent and acknowledged.
func (c *TCPConn) SetLinger(sec int) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setLinger(c.fd, sec)
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setKeepAlive(c.fd, keepalive)
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets
// (Nagle's algorithm).  The default is true (no delay), meaning
// that data is sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setNoDelay(c.fd, noDelay)
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *TCPConn) File() (f *os.File, err os.Error) { return c.fd.dup() }

// DialTCP is like Dial but can only connect to TCP networks
// and returns a TCPConn structure.
func DialTCP(net string, laddr, raddr *TCPAddr) (c *TCPConn, err os.Error) {
	if raddr == nil {
		return nil, &OpError{"dial", "tcp", nil, errMissingAddress}
	}
	fd, e := internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_STREAM, 0, "dial", sockaddrToTCP)
	if e != nil {
		return nil, e
	}
	return newTCPConn(fd), nil
}

// TCPListener is a TCP network listener.
// Clients should typically use variables of type Listener
// instead of assuming TCP.
type TCPListener struct {
	fd *netFD
}

// ListenTCP announces on the TCP address laddr and returns a TCP listener.
// Net must be "tcp", "tcp4", or "tcp6".
// If laddr has a port of 0, it means to listen on some available port.
// The caller can use l.Addr() to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (l *TCPListener, err os.Error) {
	fd, err := internetSocket(net, laddr.toAddr(), nil, syscall.SOCK_STREAM, 0, "listen", sockaddrToTCP)
	if err != nil {
		return nil, err
	}
	errno := syscall.Listen(fd.sysfd, listenBacklog())
	if errno != 0 {
		closesocket(fd.sysfd)
		return nil, &OpError{"listen", "tcp", laddr, os.Errno(errno)}
	}
	l = new(TCPListener)
	l.fd = fd
	return l, nil
}

// AcceptTCP accepts the next incoming call and returns the new connection
// and the remote address.
func (l *TCPListener) AcceptTCP() (c *TCPConn, err os.Error) {
	if l == nil || l.fd == nil || l.fd.sysfd < 0 {
		return nil, os.EINVAL
	}
	fd, err := l.fd.accept(sockaddrToTCP)
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *TCPListener) Accept() (c Conn, err os.Error) {
	c1, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	return c1, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *TCPListener) Close() os.Error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}
	return l.fd.Close()
}

// Addr returns the listener's network address, a *TCPAddr.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (l *TCPListener) File() (f *os.File, err os.Error) { return l.fd.dup() }
