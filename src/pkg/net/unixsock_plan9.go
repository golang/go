// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unix domain sockets stubs for Plan 9

package net

import (
	"os"
	"syscall"
	"time"
)

// UnixConn is an implementation of the Conn interface for connections
// to Unix domain sockets.
type UnixConn bool

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *UnixConn) Read(b []byte) (int, error) {
	return 0, syscall.EPLAN9
}

// Write implements the Conn Write method.
func (c *UnixConn) Write(b []byte) (int, error) {
	return 0, syscall.EPLAN9
}

// LocalAddr returns the local network address.
func (c *UnixConn) LocalAddr() Addr {
	return nil
}

// RemoteAddr returns the remote network address.
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

// SetReadBuffer sets the size of the operating system's receive
// buffer associated with the connection.
func (c *UnixConn) SetReadBuffer(bytes int) error {
	return syscall.EPLAN9
}

// SetWriteBuffer sets the size of the operating system's transmit
// buffer associated with the connection.
func (c *UnixConn) SetWriteBuffer(bytes int) error {
	return syscall.EPLAN9
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *UnixConn) File() (f *os.File, err error) {
	return nil, syscall.EPLAN9
}

// Close closes the Unix domain connection.
func (c *UnixConn) Close() error {
	return syscall.EPLAN9
}

// ReadFromUnix reads a packet from c, copying the payload into b.  It
// returns the number of bytes copied into b and the source address of
// the packet.
//
// ReadFromUnix can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *UnixConn) ReadFromUnix(b []byte) (int, *UnixAddr, error) {
	return 0, nil, syscall.EPLAN9
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (int, Addr, error) {
	return 0, nil, syscall.EPLAN9
}

// ReadMsgUnix reads a packet from c, copying the payload into b and
// the associated out-of-band data into oob.  It returns the number of
// bytes copied into b, the number of bytes copied into oob, the flags
// that were set on the packet, and the source address of the packet.
func (c *UnixConn) ReadMsgUnix(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	return 0, 0, 0, nil, syscall.EPLAN9
}

// WriteToUnix writes a packet to addr via c, copying the payload from b.
//
// WriteToUnix can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *UnixConn) WriteToUnix(b []byte, addr *UnixAddr) (int, error) {
	return 0, syscall.EPLAN9
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UnixConn) WriteTo(b []byte, addr Addr) (int, error) {
	return 0, syscall.EPLAN9
}

// WriteMsgUnix writes a packet to addr via c, copying the payload
// from b and the associated out-of-band data from oob.  It returns
// the number of payload and out-of-band bytes written.
func (c *UnixConn) WriteMsgUnix(b, oob []byte, addr *UnixAddr) (n, oobn int, err error) {
	return 0, 0, syscall.EPLAN9
}

// CloseRead shuts down the reading side of the Unix domain
// connection.  Most callers should just use Close.
func (c *UnixConn) CloseRead() error {
	return syscall.EPLAN9
}

// CloseWrite shuts down the writing side of the Unix domain
// connection.  Most callers should just use Close.
func (c *UnixConn) CloseWrite() error {
	return syscall.EPLAN9
}

// DialUnix connects to the remote address raddr on the network net,
// which must be "unix" or "unixgram".  If laddr is not nil, it is
// used as the local address for the connection.
func DialUnix(net string, laddr, raddr *UnixAddr) (*UnixConn, error) {
	return dialUnix(net, laddr, raddr, noDeadline)
}

func dialUnix(net string, laddr, raddr *UnixAddr, deadline time.Time) (*UnixConn, error) {
	return nil, syscall.EPLAN9
}

// UnixListener is a Unix domain socket listener.  Clients should
// typically use variables of type Listener instead of assuming Unix
// domain sockets.
type UnixListener bool

// ListenUnix announces on the Unix domain socket laddr and returns a
// Unix listener.  Net must be "unix" (stream sockets).
func ListenUnix(net string, laddr *UnixAddr) (*UnixListener, error) {
	return nil, syscall.EPLAN9
}

// AcceptUnix accepts the next incoming call and returns the new
// connection and the remote address.
func (l *UnixListener) AcceptUnix() (*UnixConn, error) {
	return nil, syscall.EPLAN9
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *UnixListener) Accept() (Conn, error) {
	return nil, syscall.EPLAN9
}

// Close stops listening on the Unix address.  Already accepted
// connections are not closed.
func (l *UnixListener) Close() error {
	return syscall.EPLAN9
}

// Addr returns the listener's network address.
func (l *UnixListener) Addr() Addr { return nil }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *UnixListener) SetDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
func (l *UnixListener) File() (*os.File, error) {
	return nil, syscall.EPLAN9
}

// ListenUnixgram listens for incoming Unix datagram packets addressed
// to the local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send UDP packets
// with per-packet addressing.  The network net must be "unixgram".
func ListenUnixgram(net string, laddr *UnixAddr) (*UDPConn, error) {
	return nil, syscall.EPLAN9
}
