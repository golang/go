// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP for Plan 9

package net

import (
	"os"
	"time"
)

// TCPConn is an implementation of the Conn interface
// for TCP network connections.
type TCPConn struct {
	plan9Conn
}

// SetDeadline implements the net.Conn SetDeadline method.
func (c *TCPConn) SetDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetReadDeadline implements the net.Conn SetReadDeadline method.
func (c *TCPConn) SetReadDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetWriteDeadline implements the net.Conn SetWriteDeadline method.
func (c *TCPConn) SetWriteDeadline(t time.Time) error {
	return os.EPLAN9
}

// CloseRead shuts down the reading side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseRead() error {
	if !c.ok() {
		return os.EINVAL
	}
	return os.EPLAN9
}

// CloseWrite shuts down the writing side of the TCP connection.
// Most callers should just use Close.
func (c *TCPConn) CloseWrite() error {
	if !c.ok() {
		return os.EINVAL
	}
	return os.EPLAN9
}

// DialTCP connects to the remote address raddr on the network net,
// which must be "tcp", "tcp4", or "tcp6".  If laddr is not nil, it is used
// as the local address for the connection.
func DialTCP(net string, laddr, raddr *TCPAddr) (c *TCPConn, err error) {
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

// TCPListener is a TCP network listener.
// Clients should typically use variables of type Listener
// instead of assuming TCP.
type TCPListener struct {
	plan9Listener
}

// ListenTCP announces on the TCP address laddr and returns a TCP listener.
// Net must be "tcp", "tcp4", or "tcp6".
// If laddr has a port of 0, it means to listen on some available port.
// The caller can use l.Addr() to retrieve the chosen address.
func ListenTCP(net string, laddr *TCPAddr) (l *TCPListener, err error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		return nil, &OpError{"listen", net, nil, errMissingAddress}
	}
	l1, err := listenPlan9(net, laddr)
	if err != nil {
		return
	}
	return &TCPListener{*l1}, nil
}
