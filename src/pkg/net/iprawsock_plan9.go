// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"time"
)

// IPConn is the implementation of the Conn and PacketConn interfaces
// for IP network connections.
type IPConn struct {
	conn
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
// associated out-of-band data into oob.  It returns the number of
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
// address laddr.  The returned connection's ReadFrom and WriteTo
// methods can be used to receive and send IP packets with per-packet
// addressing.
func ListenIP(netProto string, laddr *IPAddr) (*IPConn, error) {
	return nil, syscall.EPLAN9
}
