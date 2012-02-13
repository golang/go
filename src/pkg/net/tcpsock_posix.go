// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// TCP sockets

package net

import (
	"io"
	"os"
	"syscall"
	"time"
)

// BUG(rsc): On OpenBSD, listening on the "tcp" network does not listen for
// both IPv4 and IPv6 connections. This is due to the fact that IPv4 traffic
// will not be routed to an IPv6 socket - two separate sockets are required
// if both AFs are to be supported. See inet6(4) on OpenBSD for details.

func sockaddrToTCP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &TCPAddr{sa.Addr[0:], sa.Port}
	case *syscall.SockaddrInet6:
		return &TCPAddr{sa.Addr[0:], sa.Port}
	}
	return nil
}

func (a *TCPAddr) family() int {
	if a == nil || len(a.IP) <= IPv4len {
		return syscall.AF_INET
	}
	if a.IP.To4() != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

func (a *TCPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	return ipToSockaddr(family, a.IP, a.Port)
}

func (a *TCPAddr) toAddr() sockaddr {
	if a == nil { // nil *TCPAddr
		return nil // nil interface
	}
	return a
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

// Read implements the Conn Read method.
func (c *TCPConn) Read(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Read(b)
}

// ReadFrom implements the io.ReaderFrom ReadFrom method.
func (c *TCPConn) ReadFrom(r io.Reader) (int64, error) {
	if n, err, handled := sendFile(c.fd, r); handled {
		return n, err
	}
	return genericReadFrom(c, r)
}

// Write implements the Conn Write method.
func (c *TCPConn) Write(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the TCP connection.
func (c *TCPConn) Close() error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return os.EINVAL
	}
	return c.fd.CloseRead()
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return os.EINVAL
	}
	return c.fd.CloseWrite()
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

// SetDeadline implements the Conn SetDeadline method.
func (c *TCPConn) SetDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setDeadline(c.fd, t)
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *TCPConn) SetReadDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadDeadline(c.fd, t)
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *TCPConn) SetWriteDeadline(t time.Time) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteDeadline(c.fd, t)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *TCPConn) SetReadBuffer(bytes int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *TCPConn) SetWriteBuffer(bytes int) error {
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
func (c *TCPConn) SetLinger(sec int) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setLinger(c.fd, sec)
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setKeepAlive(c.fd, keepalive)
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets
// (Nagle's algorithm).  The default is true (no delay), meaning
// that data is sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) error {
	if !c.ok() {
		return os.EINVAL
	}
	return setNoDelay(c.fd, noDelay)
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *TCPConn) File() (f *os.File, err error) { return c.fd.dup() }

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is used
// as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	if raddr == nil {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}

	fd, err := internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_STREAM, 0, "dial", sockaddrToTCP)

	// TCP has a rarely used mechanism called a 'simultaneous connection' in
	// which Dial("tcp", addr1, addr2) run on the machine at addr1 can
	// connect to a simultaneous Dial("tcp", addr2, addr1) run on the machine
	// at addr2, without either machine executing Listen.  If laddr == nil,
	// it means we want the kernel to pick an appropriate originating local
	// address.  Some Linux kernels cycle blindly through a fixed range of
	// local ports, regardless of destination port.  If a kernel happens to
	// pick local port 50001 as the source for a Dial("tcp", "", "localhost:50001"),
	// then the Dial will succeed, having simultaneously connected to itself.
	// This can only happen when we are letting the kernel pick a port (laddr == nil)
	// and when there is no listener for the destination address.
	// It's hard to argue this is anything other than a kernel bug.  If we
	// see this happen, rather than expose the buggy effect to users, we
	// close the fd and try again.  If it happens twice more, we relent and
	// use the result.  See also:
	//	http://golang.org/issue/2690
	//	http://stackoverflow.com/questions/4949858/
	for i := 0; i < 2 && err == nil && laddr == nil && selfConnect(fd); i++ {
		fd.Close()
		fd, err = internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_STREAM, 0, "dial", sockaddrToTCP)
	}

	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

func selfConnect(fd *netFD) bool {
	l := fd.laddr.(*TCPAddr)
	r := fd.raddr.(*TCPAddr)
	return l.Port == r.Port && l.IP.Equal(r.IP)
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
func ListenTCP(net string, laddr *TCPAddr) (*TCPListener, error) {
	fd, err := internetSocket(net, laddr.toAddr(), nil, syscall.SOCK_STREAM, 0, "listen", sockaddrToTCP)
	if err != nil {
		return nil, err
	}
	err = syscall.Listen(fd.sysfd, listenerBacklog)
	if err != nil {
		closesocket(fd.sysfd)
		return nil, &OpError{"listen", net, laddr, err}
	}
	l := new(TCPListener)
	l.fd = fd
	return l, nil
}

// AcceptTCP accepts the next incoming call and returns the new connection
// and the remote address.
func (l *TCPListener) AcceptTCP() (c *TCPConn, err error) {
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
func (l *TCPListener) Accept() (c Conn, err error) {
	c1, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	return c1, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *TCPListener) Close() error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}
	return l.fd.Close()
}

// Addr returns the listener's network address, a *TCPAddr.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *TCPListener) SetDeadline(t time.Time) error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}
	return setDeadline(l.fd, t)
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (l *TCPListener) File() (f *os.File, err error) { return l.fd.dup() }
