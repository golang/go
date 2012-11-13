// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP sockets for Plan 9

package net

import (
	"syscall"
	"time"
)

// TCPConn is an implementation of the Conn interface for TCP network
// connections.
type TCPConn struct {
	plan9Conn
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (c *TCPConn, err error) {
	return dialTCP(net, laddr, raddr, noDeadline)
}

func dialTCP(net string, laddr, raddr *TCPAddr, deadline time.Time) (c *TCPConn, err error) {
	if !deadline.IsZero() {
		panic("net.dialTCP: deadline not implemented on Plan 9")
	}
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}
	c1, err := dialPlan9(net, laddr, raddr)
	if err != nil {
		return
	}
	return &TCPConn{*c1}, nil
}

// TCPListener is a TCP network listener.  Clients should typically
// use variables of type Listener instead of assuming TCP.
type TCPListener struct {
	plan9Listener
}

func (l *TCPListener) Close() error {
	if l == nil || l.ctl == nil {
		return syscall.EINVAL
	}
	if _, err := l.ctl.WriteString("hangup"); err != nil {
		l.ctl.Close()
		return err
	}
	return l.ctl.Close()
}

// ListenTCP announces on the TCP address laddr and returns a TCP
// listener.  Net must be "tcp", "tcp4", or "tcp6".  If laddr has a
// port of 0, it means to listen on some available port.  The caller
// can use l.Addr() to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (l *TCPListener, err error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		laddr = &TCPAddr{}
	}
	l1, err := listenPlan9(net, laddr)
	if err != nil {
		return
	}
	return &TCPListener{*l1}, nil
}
