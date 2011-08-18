// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// UDP for Plan 9

package net

import (
	"os"
)

// UDPConn is the implementation of the Conn and PacketConn
// interfaces for UDP network connections.
type UDPConn struct {
	plan9Conn
}

// UDP-specific methods.

// ReadFromUDP reads a UDP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUDP can be made to time out and return an error with Timeout() == true
// after a fixed time limit; see SetTimeout and SetReadTimeout.
func (c *UDPConn) ReadFromUDP(b []byte) (n int, addr *UDPAddr, err os.Error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	if c.data == nil {
		c.data, err = os.OpenFile(c.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, nil, err
		}
	}
	buf := make([]byte, udpHeaderSize+len(b))
	m, err := c.data.Read(buf)
	if err != nil {
		return
	}
	if m < udpHeaderSize {
		return 0, nil, os.NewError("short read reading UDP header")
	}
	buf = buf[:m]

	h, buf := unmarshalUDPHeader(buf)
	n = copy(b, buf)
	return n, &UDPAddr{h.raddr, int(h.rport)}, nil
}

// ReadFrom implements the net.PacketConn ReadFrom method.
func (c *UDPConn) ReadFrom(b []byte) (n int, addr Addr, err os.Error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	return c.ReadFromUDP(b)
}

// WriteToUDP writes a UDP packet to addr via c, copying the payload from b.
//
// WriteToUDP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetTimeout and SetWriteTimeout.
// On packet-oriented connections, write timeouts are rare.
func (c *UDPConn) WriteToUDP(b []byte, addr *UDPAddr) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	if c.data == nil {
		c.data, err = os.OpenFile(c.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
	}
	h := new(udpHeader)
	h.raddr = addr.IP.To16()
	h.laddr = c.laddr.(*UDPAddr).IP.To16()
	h.ifcaddr = IPv6zero // ignored (receive only)
	h.rport = uint16(addr.Port)
	h.lport = uint16(c.laddr.(*UDPAddr).Port)

	buf := make([]byte, udpHeaderSize+len(b))
	i := copy(buf, h.Bytes())
	copy(buf[i:], b)
	return c.data.Write(buf)
}

// WriteTo implements the net.PacketConn WriteTo method.
func (c *UDPConn) WriteTo(b []byte, addr Addr) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	a, ok := addr.(*UDPAddr)
	if !ok {
		return 0, &OpError{"writeto", "udp", addr, os.EINVAL}
	}
	return c.WriteToUDP(b, a)
}

// DialUDP connects to the remote address raddr on the network net,
// which must be "udp", "udp4", or "udp6".  If laddr is not nil, it is used
// as the local address for the connection.
func DialUDP(net string, laddr, raddr *UDPAddr) (c *UDPConn, err os.Error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", "udp", nil, errMissingAddress}
	}
	c1, err := dialPlan9(net, laddr, raddr)
	if err != nil {
		return
	}
	return &UDPConn{*c1}, nil
}

const udpHeaderSize = 16*3 + 2*2

type udpHeader struct {
	raddr, laddr, ifcaddr IP
	rport, lport          uint16
}

func (h *udpHeader) Bytes() []byte {
	b := make([]byte, udpHeaderSize)
	i := 0
	i += copy(b[i:i+16], h.raddr)
	i += copy(b[i:i+16], h.laddr)
	i += copy(b[i:i+16], h.ifcaddr)
	b[i], b[i+1], i = byte(h.rport>>8), byte(h.rport), i+2
	b[i], b[i+1], i = byte(h.lport>>8), byte(h.lport), i+2
	return b
}

func unmarshalUDPHeader(b []byte) (*udpHeader, []byte) {
	h := new(udpHeader)
	h.raddr, b = IP(b[:16]), b[16:]
	h.laddr, b = IP(b[:16]), b[16:]
	h.ifcaddr, b = IP(b[:16]), b[16:]
	h.rport, b = uint16(b[0])<<8|uint16(b[1]), b[2:]
	h.lport, b = uint16(b[0])<<8|uint16(b[1]), b[2:]
	return h, b
}

// ListenUDP listens for incoming UDP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send UDP
// packets with per-packet addressing.
func ListenUDP(net string, laddr *UDPAddr) (c *UDPConn, err os.Error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		return nil, &OpError{"listen", "udp", nil, errMissingAddress}
	}
	l, err := listenPlan9(net, laddr)
	if err != nil {
		return
	}
	_, err = l.ctl.WriteString("headers")
	if err != nil {
		return
	}
	return &UDPConn{*l.plan9Conn()}, nil
}

// JoinGroup joins the IP multicast group named by addr on ifi,
// which specifies the interface to join.  JoinGroup uses the
// default multicast interface if ifi is nil.
func (c *UDPConn) JoinGroup(ifi *Interface, addr IP) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return os.EPLAN9
}

// LeaveGroup exits the IP multicast group named by addr on ifi.
func (c *UDPConn) LeaveGroup(ifi *Interface, addr IP) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return os.EPLAN9
}
