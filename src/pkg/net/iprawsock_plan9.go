// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// (Raw) IP sockets stubs for Plan 9

package net

import (
	"os"
)

// IPConn is the implementation of the Conn and PacketConn
// interfaces for IP network connections.
type IPConn bool

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the net.Conn Read method.
func (c *IPConn) Read(b []byte) (n int, err os.Error) {
	return 0, os.EPLAN9
}

// Write implements the net.Conn Write method.
func (c *IPConn) Write(b []byte) (n int, err os.Error) {
	return 0, os.EPLAN9
}

// Close closes the IP connection.
func (c *IPConn) Close() os.Error {
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

// SetTimeout implements the net.Conn SetTimeout method.
func (c *IPConn) SetTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// SetReadTimeout implements the net.Conn SetReadTimeout method.
func (c *IPConn) SetReadTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// SetWriteTimeout implements the net.Conn SetWriteTimeout method.
func (c *IPConn) SetWriteTimeout(nsec int64) os.Error {
	return os.EPLAN9
}

// IP-specific methods.

// ReadFrom implements the net.PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (n int, addr Addr, err os.Error) {
	err = os.EPLAN9
	return
}

// WriteToIP writes a IP packet to addr via c, copying the payload from b.
//
// WriteToIP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetTimeout and SetWriteTimeout.
// On packet-oriented connections, write timeouts are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (n int, err os.Error) {
	return 0, os.EPLAN9
}

// WriteTo implements the net.PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (n int, err os.Error) {
	return 0, os.EPLAN9
}

func splitNetProto(netProto string) (net string, proto int, err os.Error) {
	err = os.EPLAN9
	return
}

// DialIP connects to the remote address raddr on the network net,
// which must be "ip", "ip4", or "ip6".
func DialIP(netProto string, laddr, raddr *IPAddr) (c *IPConn, err os.Error) {
	return nil, os.EPLAN9
}

// ListenIP listens for incoming IP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send IP
// packets with per-packet addressing.
func ListenIP(netProto string, laddr *IPAddr) (c *IPConn, err os.Error) {
	return nil, os.EPLAN9
}
