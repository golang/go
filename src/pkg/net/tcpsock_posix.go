// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

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
		return &TCPAddr{IP: sa.Addr[0:], Port: sa.Port}
	case *syscall.SockaddrInet6:
		return &TCPAddr{IP: sa.Addr[0:], Port: sa.Port, Zone: zoneToString(int(sa.ZoneId))}
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

func (a *TCPAddr) isWildcard() bool {
	if a == nil || a.IP == nil {
		return true
	}
	return a.IP.IsUnspecified()
}

func (a *TCPAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return ipToSockaddr(family, a.IP, a.Port, a.Zone)
}

// TCPConn is an implementation of the Conn interface for TCP network
// connections.
type TCPConn struct {
	conn
}

func newTCPConn(fd *netFD) *TCPConn {
	c := &TCPConn{conn{fd}}
	c.SetNoDelay(true)
	return c
}

// ReadFrom implements the io.ReaderFrom ReadFrom method.
func (c *TCPConn) ReadFrom(r io.Reader) (int64, error) {
	if n, err, handled := sendFile(c.fd, r); handled {
		return n, err
	}
	return genericReadFrom(c, r)
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.CloseRead()
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.CloseWrite()
}

// SetLinger sets the behavior of Close() on a connection which still
// has data waiting to be sent or to be acknowledged.
//
// If sec < 0 (the default), Close returns immediately and the
// operating system finishes sending the data in the background.
//
// If sec == 0, Close returns immediately and the operating system
// discards any unsent or unacknowledged data.
//
// If sec > 0, Close blocks for at most sec seconds waiting for data
// to be sent and acknowledged.
func (c *TCPConn) SetLinger(sec int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setLinger(c.fd, sec)
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setKeepAlive(c.fd, keepalive)
}

// SetKeepAlivePeriod sets period between keep alives.
func (c *TCPConn) SetKeepAlivePeriod(d time.Duration) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setKeepAlivePeriod(c.fd, d)
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets (Nagle's
// algorithm).  The default is true (no delay), meaning that data is
// sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setNoDelay(c.fd, noDelay)
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: UnknownNetworkError(net)}
	}
	if raddr == nil {
		return nil, &OpError{Op: "dial", Net: net, Addr: nil, Err: errMissingAddress}
	}
	return dialTCP(net, laddr, raddr, noDeadline)
}

func dialTCP(net string, laddr, raddr *TCPAddr, deadline time.Time) (*TCPConn, error) {
	fd, err := internetSocket(net, laddr, raddr, deadline, syscall.SOCK_STREAM, 0, "dial", sockaddrToTCP)

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
	//
	// The opposite can also happen: if we ask the kernel to pick an appropriate
	// originating local address, sometimes it picks one that is already in use.
	// So if the error is EADDRNOTAVAIL, we have to try again too, just for
	// a different reason.
	//
	// The kernel socket code is no doubt enjoying watching us squirm.
	for i := 0; i < 2 && (laddr == nil || laddr.Port == 0) && (selfConnect(fd, err) || spuriousENOTAVAIL(err)); i++ {
		if err == nil {
			fd.Close()
		}
		fd, err = internetSocket(net, laddr, raddr, deadline, syscall.SOCK_STREAM, 0, "dial", sockaddrToTCP)
	}

	if err != nil {
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: err}
	}
	return newTCPConn(fd), nil
}

func selfConnect(fd *netFD, err error) bool {
	// If the connect failed, we clearly didn't connect to ourselves.
	if err != nil {
		return false
	}

	// The socket constructor can return an fd with raddr nil under certain
	// unknown conditions. The errors in the calls there to Getpeername
	// are discarded, but we can't catch the problem there because those
	// calls are sometimes legally erroneous with a "socket not connected".
	// Since this code (selfConnect) is already trying to work around
	// a problem, we make sure if this happens we recognize trouble and
	// ask the DialTCP routine to try again.
	// TODO: try to understand what's really going on.
	if fd.laddr == nil || fd.raddr == nil {
		return true
	}
	l := fd.laddr.(*TCPAddr)
	r := fd.raddr.(*TCPAddr)
	return l.Port == r.Port && l.IP.Equal(r.IP)
}

func spuriousENOTAVAIL(err error) bool {
	e, ok := err.(*OpError)
	return ok && e.Err == syscall.EADDRNOTAVAIL
}

// TCPListener is a TCP network listener.  Clients should typically
// use variables of type Listener instead of assuming TCP.
type TCPListener struct {
	fd *netFD
}

// AcceptTCP accepts the next incoming call and returns the new
// connection.
func (l *TCPListener) AcceptTCP() (*TCPConn, error) {
	if l == nil || l.fd == nil {
		return nil, syscall.EINVAL
	}
	fd, err := l.fd.accept(sockaddrToTCP)
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *TCPListener) Accept() (Conn, error) {
	c, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	return c, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *TCPListener) Close() error {
	if l == nil || l.fd == nil {
		return syscall.EINVAL
	}
	return l.fd.Close()
}

// Addr returns the listener's network address, a *TCPAddr.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *TCPListener) SetDeadline(t time.Time) error {
	if l == nil || l.fd == nil {
		return syscall.EINVAL
	}
	return l.fd.setDeadline(t)
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's.  Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *TCPListener) File() (f *os.File, err error) { return l.fd.dup() }

// ListenTCP announces on the TCP address laddr and returns a TCP
// listener.  Net must be "tcp", "tcp4", or "tcp6".  If laddr has a
// port of 0, ListenTCP will choose an available port.  The caller can
// use the Addr method of TCPListener to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (*TCPListener, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: UnknownNetworkError(net)}
	}
	if laddr == nil {
		laddr = &TCPAddr{}
	}
	fd, err := internetSocket(net, laddr, nil, noDeadline, syscall.SOCK_STREAM, 0, "listen", sockaddrToTCP)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: err}
	}
	return &TCPListener{fd}, nil
}
