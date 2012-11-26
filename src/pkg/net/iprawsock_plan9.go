// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Raw IP sockets for Plan 9

package net

import (
	"os"
	"syscall"
	"time"
)

// IPConn is the implementation of the Conn and PacketConn interfaces
// for IP network connections.
type IPConn bool

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *IPConn) Read(b []byte) (int, error) {
	return 0, syscall.EPLAN9
}

// Write implements the Conn Write method.
func (c *IPConn) Write(b []byte) (int, error) {
	return 0, syscall.EPLAN9
}

// LocalAddr returns the local network address.
func (c *IPConn) LocalAddr() Addr {
	return nil
}

// RemoteAddr returns the remote network address.
func (c *IPConn) RemoteAddr() Addr {
	return nil
}

// SetDeadline implements the Conn SetDeadline method.
func (c *IPConn) SetDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *IPConn) SetReadDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *IPConn) SetWriteDeadline(t time.Time) error {
	return syscall.EPLAN9
}

// SetReadBuffer sets the size of the operating system's receive
// buffer associated with the connection.
func (c *IPConn) SetReadBuffer(bytes int) error {
	return syscall.EPLAN9
}

// SetWriteBuffer sets the size of the operating system's transmit
// buffer associated with the connection.
func (c *IPConn) SetWriteBuffer(bytes int) error {
	return syscall.EPLAN9
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *IPConn) File() (f *os.File, err error) {
	return nil, syscall.EPLAN9
}

// Close closes the IP connection.
func (c *IPConn) Close() error {
	return syscall.EPLAN9
}

// ReadFromIP reads an IP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *IPConn) ReadFromIP(b []byte) (int, *IPAddr, error) {
	return 0, nil, syscall.EPLAN9
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (int, Addr, error) {
	return 0, nil, syscall.EPLAN9
}

// ReadMsgIP reads a packet from c, copying the payload into b and the
// associdated out-of-band data into oob.  It returns the number of
// bytes copied into b, the number of bytes copied into oob, the flags
// that were set on the packet and the source address of the packet.
func (c *IPConn) ReadMsgIP(b, oob []byte) (n, oobn, flags int, addr *IPAddr, err error) {
	return 0, 0, 0, nil, syscall.EPLAN9
}

// WriteToIP writes an IP packet to addr via c, copying the payload
// from b.
//
// WriteToIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (int, error) {
	return 0, syscall.EPLAN9
}

// WriteTo implements the PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (int, error) {
	return 0, syscall.EPLAN9
}

// WriteMsgIP writes a packet to addr via c, copying the payload from
// b and the associated out-of-band data from oob.  It returns the
// number of payload and out-of-band bytes written.
func (c *IPConn) WriteMsgIP(b, oob []byte, addr *IPAddr) (n, oobn int, err error) {
	return 0, 0, syscall.EPLAN9
}

// DialIP connects to the remote address raddr on the network protocol
// netProto, which must be "ip", "ip4", or "ip6" followed by a colon
// and a protocol number or name.
func DialIP(netProto string, laddr, raddr *IPAddr) (*IPConn, error) {
	return dialIP(netProto, laddr, raddr, noDeadline)
}

func dialIP(netProto string, laddr, raddr *IPAddr, deadline time.Time) (*IPConn, error) {
	return nil, syscall.EPLAN9
}

// ListenIP listens for incoming IP packets addressed to the local
// address laddr.  The returned connection c's ReadFrom and WriteTo
// methods can be used to receive and send IP packets with per-packet
// addressing.
func ListenIP(netProto string, laddr *IPAddr) (*IPConn, error) {
	return nil, syscall.EPLAN9
}
