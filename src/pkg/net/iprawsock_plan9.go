// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// (Raw) IP sockets stubs for Plan 9

package net

import (
	"os"
	"time"
)

// IPConn is the implementation of the Conn and PacketConn
// interfaces for IP network connections.
type IPConn bool

// SetDeadline implements the Conn SetDeadline method.
func (c *IPConn) SetDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *IPConn) SetReadDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *IPConn) SetWriteDeadline(t time.Time) error {
	return os.EPLAN9
}

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *IPConn) Read(b []byte) (int, error) {
	return 0, os.EPLAN9
}

// Write implements the Conn Write method.
func (c *IPConn) Write(b []byte) (int, error) {
	return 0, os.EPLAN9
}

// Close closes the IP connection.
func (c *IPConn) Close() error {
	return os.EPLAN9
}

// LocalAddr returns the local network address.
func (c *IPConn) LocalAddr() Addr {
	return nil
}

// RemoteAddr returns the remote network address, a *IPAddr.
func (c *IPConn) RemoteAddr() Addr {
	return nil
}

// IP-specific methods.

// ReadFromIP reads a IP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *IPConn) ReadFromIP(b []byte) (int, *IPAddr, error) {
	return 0, nil, os.EPLAN9
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (int, Addr, error) {
	return 0, nil, os.EPLAN9
}

// WriteToIP writes a IP packet to addr via c, copying the payload from b.
//
// WriteToIP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetDeadline and SetWriteDeadline.
// On packet-oriented connections, write timeouts are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (int, error) {
	return 0, os.EPLAN9
}

// WriteTo implements the PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (int, error) {
	return 0, os.EPLAN9
}

// DialIP connects to the remote address raddr on the network protocol netProto,
// which must be "ip", "ip4", or "ip6" followed by a colon and a protocol number or name.
func DialIP(netProto string, laddr, raddr *IPAddr) (*IPConn, error) {
	return nil, os.EPLAN9
}

// ListenIP listens for incoming IP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send IP
// packets with per-packet addressing.
func ListenIP(netProto string, laddr *IPAddr) (*IPConn, error) {
	return nil, os.EPLAN9
}
