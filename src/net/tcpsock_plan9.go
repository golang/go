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
	n, err := genericReadFrom(c, r)
	if err != nil && err != io.EOF {
		err = &OpError{Op: "read", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.closeRead()
	if err != nil {
		err = &OpError{Op: "close", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return err
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.closeWrite()
	if err != nil {
		err = &OpError{Op: "close", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return err
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
	return &OpError{Op: "set", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: syscall.EPLAN9}
}

// SetKeepAlive sets whether the operating system should send
// keepalive messages on the connection.
func (c *TCPConn) SetKeepAlive(keepalive bool) error {
	if !c.ok() {
		return syscall.EPLAN9
	}
	if err := setKeepAlive(c.fd, keepalive); err != nil {
		return &OpError{Op: "set", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetKeepAlivePeriod sets period between keep alives.
func (c *TCPConn) SetKeepAlivePeriod(d time.Duration) error {
	if !c.ok() {
		return syscall.EPLAN9
	}
	if err := setKeepAlivePeriod(c.fd, d); err != nil {
		return &OpError{Op: "set", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// SetNoDelay controls whether the operating system should delay
// packet transmission in hopes of sending fewer packets (Nagle's
// algorithm).  The default is true (no delay), meaning that data is
// sent as soon as possible after a Write.
func (c *TCPConn) SetNoDelay(noDelay bool) error {
	return &OpError{Op: "set", Net: c.fd.dir, Source: c.fd.laddr, Addr: c.fd.raddr, Err: syscall.EPLAN9}
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	return dialTCP(net, laddr, raddr, noDeadline, noCancel)
}

func dialTCP(net string, laddr, raddr *TCPAddr, deadline time.Time, cancel <-chan struct{}) (*TCPConn, error) {
	if !deadline.IsZero() {
		panic("net.dialTCP: deadline not implemented on Plan 9")
	}
	// TODO(bradfitz,0intro): also use the cancel channel.
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, &OpError{Op: "dial", Net: net, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: UnknownNetworkError(net)}
	}
	if raddr == nil {
		return nil, &OpError{Op: "dial", Net: net, Source: laddr.opAddr(), Addr: nil, Err: errMissingAddress}
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
// connection.
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
		return &OpError{Op: "close", Net: l.fd.dir, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	err := l.fd.ctl.Close()
	if err != nil {
		err = &OpError{Op: "close", Net: l.fd.dir, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return err
}

// Addr returns the listener's network address, a *TCPAddr.
// The Addr returned is shared by all invocations of Addr, so
// do not modify it.
func (l *TCPListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *TCPListener) SetDeadline(t time.Time) error {
	if l == nil || l.fd == nil || l.fd.ctl == nil {
		return syscall.EINVAL
	}
	if err := l.fd.setDeadline(t); err != nil {
		return &OpError{Op: "set", Net: l.fd.dir, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return nil
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's.  Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *TCPListener) File() (f *os.File, err error) {
	f, err = l.dup()
	if err != nil {
		err = &OpError{Op: "file", Net: l.fd.dir, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return
}

// ListenTCP announces on the TCP address laddr and returns a TCP
// listener.  Net must be "tcp", "tcp4", or "tcp6".  If laddr has a
// port of 0, ListenTCP will choose an available port.  The caller can
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
	fd, err := listenPlan9(net, laddr)
	if err != nil {
		return nil, err
	}
	return &TCPListener{fd}, nil
}
