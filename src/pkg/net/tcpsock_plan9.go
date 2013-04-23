// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
	"time"
)

// TCPConn is an implementation of the Conn interface for TCP network
// connections.
type TCPConn struct {
	conn
}

func newTCPConn(fd *netFD) *TCPConn {
	return &TCPConn{conn{fd}}
}

// ReadFrom implements the io.ReaderFrom ReadFrom method.
func (c *TCPConn) ReadFrom(r io.Reader) (int64, error) {
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
	return syscall.EPLAN9
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) error {
	return syscall.EPLAN9
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets (Nagle's
// algorithm).  The default is true (no delay), meaning that data is
// sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) error {
	return syscall.EPLAN9
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	return dialTCP(net, laddr, raddr, noDeadline)
}

func dialTCP(net string, laddr, raddr *TCPAddr, deadline time.Time) (*TCPConn, error) {
	if !deadline.IsZero() {
		panic("net.dialTCP: deadline not implemented on Plan 9")
	}
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{"dial", net, raddr, UnknownNetworkError(net)}
	}
	if raddr == nil {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}
	fd, err := dialPlan9(net, laddr, raddr)
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

// TCPListener is a TCP network listener.  Clients should typically
// use variables of type Listener instead of assuming TCP.
type TCPListener struct {
	fd *netFD
}

// AcceptTCP accepts the next incoming call and returns the new
// connection and the remote address.
func (l *TCPListener) AcceptTCP() (*TCPConn, error) {
	if l == nil || l.fd == nil || l.fd.ctl == nil {
		return nil, syscall.EINVAL
	}
	fd, err := l.fd.acceptPlan9()
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *TCPListener) Accept() (Conn, error) {
	if l == nil || l.fd == nil || l.fd.ctl == nil {
		return nil, syscall.EINVAL
	}
	c, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	return c, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *TCPListener) Close() error {
	if l == nil || l.fd == nil || l.fd.ctl == nil {
		return syscall.EINVAL
	}
	if _, err := l.fd.ctl.WriteString("hangup"); err != nil {
		l.fd.ctl.Close()
		return &OpError{"close", l.fd.ctl.Name(), l.fd.laddr, err}
	}
	return l.fd.ctl.Close()
}

// Addr returns the listener's network address, a *TCPAddr.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *TCPListener) SetDeadline(t time.Time) error {
	if l == nil || l.fd == nil || l.fd.ctl == nil {
		return syscall.EINVAL
	}
	return setDeadline(l.fd, t)
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's.  Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *TCPListener) File() (f *os.File, err error) { return l.dup() }

// ListenTCP announces on the TCP address laddr and returns a TCP
// listener.  Net must be "tcp", "tcp4", or "tcp6".  If laddr has a
// port of 0, ListenTCP will choose an available port.  The caller can
// use the Addr method of TCPListener to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (*TCPListener, error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{"listen", net, laddr, UnknownNetworkError(net)}
	}
	if laddr == nil {
		laddr = &TCPAddr{}
	}
	fd, err := listenPlan9(net, laddr)
	if err != nil {
		return nil, err
	}
	return &TCPListener{fd}, nil
}
