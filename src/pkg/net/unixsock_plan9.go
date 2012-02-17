// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unix domain sockets stubs for Plan 9

package net

import (
	"syscall"
	"time"
)

// UnixConn is an implementation of the Conn interface
// for connections to Unix domain sockets.
type UnixConn bool

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *UnixConn) Read(b []byte) (n int, err error) {
	return 0, syscall.EPLAN9
}

// Write implements the Conn Write method.
func (c *UnixConn) Write(b []byte) (n int, err error) {
	return 0, syscall.EPLAN9
}

// Close closes the Unix domain connection.
func (c *UnixConn) Close() error {
	return syscall.EPLAN9
}

// LocalAddr returns the local network address, a *UnixAddr.
// Unlike in other protocols, LocalAddr is usually nil for dialed connections.
func (c *UnixConn) LocalAddr() Addr {
	return nil
}

// RemoteAddr returns the remote network address, a *UnixAddr.
// Unlike in other protocols, RemoteAddr is usually nil for connections
// accepted by a listener.
func (c *UnixConn) RemoteAddr() Addr {
	return nil
}

// SetDeadline implements the Conn SetDeadline method.
func (c *UnixConn) SetDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *UnixConn) SetReadDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *UnixConn) SetWriteDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (n int, addr Addr, err error) {
	err = syscall.EPLAN9
	return
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UnixConn) WriteTo(b []byte, addr Addr) (n int, err error) {
	err = syscall.EPLAN9
	return
}

// DialUnix connects to the remote address raddr on the network net,
// which must be "unix" or "unixgram".  If laddr is not nil, it is used
// as the local address for the connection.
func DialUnix(net string, laddr, raddr *UnixAddr) (c *UnixConn, err error) {
	return nil, syscall.EPLAN9
}

// UnixListener is a Unix domain socket listener.
// Clients should typically use variables of type Listener
// instead of assuming Unix domain sockets.
type UnixListener bool

// ListenUnix announces on the Unix domain socket laddr and returns a Unix listener.
// Net must be "unix" (stream sockets).
func ListenUnix(net string, laddr *UnixAddr) (l *UnixListener, err error) {
	return nil, syscall.EPLAN9
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *UnixListener) Accept() (c Conn, err error) {
	return nil, syscall.EPLAN9
}

// Close stops listening on the Unix address.
// Already accepted connections are not closed.
func (l *UnixListener) Close() error {
	return syscall.EPLAN9
}

// Addr returns the listener's network address.
func (l *UnixListener) Addr() Addr { return nil }
