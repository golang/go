// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"io"
	"os"
	"syscall"
	"time"
)

// BUG(mikio): On Windows, the File method of TCPListener is not
// implemented.

// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
	IP   IP
	Port int
	Zone string // IPv6 scoped addressing zone
}

// Network returns the address's network name, "tcp".
func (a *TCPAddr) Network() string { return "tcp" }

func (a *TCPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	ip := ipEmptyString(a.IP)
	if a.Zone != "" {
		return JoinHostPort(ip+"%"+a.Zone, itoa(a.Port))
	}
	return JoinHostPort(ip, itoa(a.Port))
}

func (a *TCPAddr) isWildcard() bool {
	if a == nil || a.IP == nil {
		return true
	}
	return a.IP.IsUnspecified()
}

func (a *TCPAddr) opAddr() Addr {
	if a == nil {
		return nil
	}
	return a
}

// ResolveTCPAddr parses addr as a TCP address of the form "host:port"
// or "[ipv6-host%zone]:port" and resolves a pair of domain name and
// port name on the network net, which must be "tcp", "tcp4" or
// "tcp6".  A literal address or host name for IPv6 must be enclosed
// in square brackets, as in "[::1]:80", "[ipv6-host]:http" or
// "[ipv6-host%zone]:80".
//
// Resolving a hostname is not recommended because this returns at most
// one of its IP addresses.
func ResolveTCPAddr(net, addr string) (*TCPAddr, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	case "": // a hint wildcard for Go 1.0 undocumented behavior
		net = "tcp"
	default:
		return nil, UnknownNetworkError(net)
	}
	addrs, err := DefaultResolver.internetAddrList(context.Background(), net, addr)
	if err != nil {
		return nil, err
	}
	return addrs.first(isIPv4).(*TCPAddr), nil
}

// TCPConn is an implementation of the Conn interface for TCP network
// connections.
type TCPConn struct {
	conn
}

// ReadFrom implements the io.ReaderFrom ReadFrom method.
func (c *TCPConn) ReadFrom(r io.Reader) (int64, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.readFrom(r)
	if err != nil && err != io.EOF {
		err = &OpError{Op: "readfrom", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := c.fd.closeRead(); err != nil {
		return &OpError{Op: "close", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := c.fd.closeWrite(); err != nil {
		return &OpError{Op: "close", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetLinger sets the behavior of Close on a connection which still
// has data waiting to be sent or to be acknowledged.
//
// If sec < 0 (the default), the operating system finishes sending the
// data in the background.
//
// If sec == 0, the operating system discards any unsent or
// unacknowledged data.
//
// If sec > 0, the data is sent in the background as with sec < 0. On
// some operating systems after sec seconds have elapsed any remaining
// unsent data may be discarded.
func (c *TCPConn) SetLinger(sec int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := setLinger(c.fd, sec); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := setKeepAlive(c.fd, keepalive); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetKeepAlivePeriod sets period between keep alives.
func (c *TCPConn) SetKeepAlivePeriod(d time.Duration) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := setKeepAlivePeriod(c.fd, d); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets (Nagle's
// algorithm).  The default is true (no delay), meaning that data is
// sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := setNoDelay(c.fd, noDelay); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

func newTCPConn(fd *netFD) *TCPConn {
	c := &TCPConn{conn{fd}}
	setNoDelay(c.fd, true)
	return c
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{Op: "dial", Net: net, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: UnknownNetworkError(net)}
	}
	if raddr == nil {
		return nil, &OpError{Op: "dial", Net: net, Source: laddr.opAddr(), Addr: nil, Err: errMissingAddress}
	}
	c, err := dialTCP(context.Background(), net, laddr, raddr)
	if err != nil {
		return nil, &OpError{Op: "dial", Net: net, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: err}
	}
	return c, nil
}

// TCPListener is a TCP network listener. Clients should typically
// use variables of type Listener instead of assuming TCP.
type TCPListener struct {
	fd *netFD
}

// AcceptTCP accepts the next incoming call and returns the new
// connection.
func (l *TCPListener) AcceptTCP() (*TCPConn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	c, err := l.accept()
	if err != nil {
		return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return c, nil
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *TCPListener) Accept() (Conn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	c, err := l.accept()
	if err != nil {
		return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return c, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *TCPListener) Close() error {
	if !l.ok() {
		return syscall.EINVAL
	}
	if err := l.close(); err != nil {
		return &OpError{Op: "close", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return nil
}

// Addr returns the listener's network address, a *TCPAddr.
// The Addr returned is shared by all invocations of Addr, so
// do not modify it.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *TCPListener) SetDeadline(t time.Time) error {
	if !l.ok() {
		return syscall.EINVAL
	}
	if err := l.fd.setDeadline(t); err != nil {
		return &OpError{Op: "set", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return nil
}

// File returns a copy of the underlying os.File, set to blocking
// mode. It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's. Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *TCPListener) File() (f *os.File, err error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	f, err = l.file()
	if err != nil {
		return nil, &OpError{Op: "file", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return
}

// ListenTCP announces on the TCP address laddr and returns a TCP
// listener. Net must be "tcp", "tcp4", or "tcp6".  If laddr has a
// port of 0, ListenTCP will choose an available port. The caller can
// use the Addr method of TCPListener to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (*TCPListener, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: laddr.opAddr(), Err: UnknownNetworkError(net)}
	}
	if laddr == nil {
		laddr = &TCPAddr{}
	}
	ln, err := listenTCP(context.Background(), net, laddr)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: laddr.opAddr(), Err: err}
	}
	return ln, nil
}
