// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unix domain sockets stubs for Plan 9

package net

import (
	"os"
)

// UnixConn is an implementation of the Conn interface
// for connections to Unix domain sockets.
type UnixConn bool

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the net.Conn Read method.
func (c *UnixConn) Read(b []byte) (n int, err os.Error) {
	return 0, os.EPLAN9
}

// Write implements the net.Conn Write method.
func (c *UnixConn) Write(b []byte) (n int, err os.Error) {
	return 0, os.EPLAN9
}

// Close closes the Unix domain connection.
func (c *UnixConn) Close() os.Error {
	return os.EPLAN9
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

// SetTimeout implements the net.Conn SetTimeout method.
func (c *UnixConn) SetTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// SetReadTimeout implements the net.Conn SetReadTimeout method.
func (c *UnixConn) SetReadTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// SetWriteTimeout implements the net.Conn SetWriteTimeout method.
func (c *UnixConn) SetWriteTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// ReadFrom implements the net.PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (n int, addr Addr, err os.Error) {
	err = os.EPLAN9
	return
}

// WriteTo implements the net.PacketConn WriteTo method.
func (c *UnixConn) WriteTo(b []byte, addr Addr) (n int, err os.Error) {
	err = os.EPLAN9
	return
}

// DialUnix connects to the remote address raddr on the network net,
// which must be "unix" or "unixgram".  If laddr is not nil, it is used
// as the local address for the connection.
func DialUnix(net string, laddr, raddr *UnixAddr) (c *UnixConn, err os.Error) {
	return nil, os.EPLAN9
}

// UnixListener is a Unix domain socket listener.
// Clients should typically use variables of type Listener
// instead of assuming Unix domain sockets.
type UnixListener bool

// ListenUnix announces on the Unix domain socket laddr and returns a Unix listener.
// Net must be "unix" (stream sockets).
func ListenUnix(net string, laddr *UnixAddr) (l *UnixListener, err os.Error) {
	return nil, os.EPLAN9
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *UnixListener) Accept() (c Conn, err os.Error) {
	return nil, os.EPLAN9
}

// Close stops listening on the Unix address.
// Already accepted connections are not closed.
func (l *UnixListener) Close() os.Error {
	return os.EPLAN9
}

// Addr returns the listener's network address.
func (l *UnixListener) Addr() Addr { return nil }
