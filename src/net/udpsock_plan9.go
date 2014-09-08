// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"os"
	"syscall"
	"time"
)

// UDPConn is the implementation of the Conn and PacketConn interfaces
// for UDP network connections.
type UDPConn struct {
	conn
}

func newUDPConn(fd *netFD) *UDPConn { return &UDPConn{conn{fd}} }

// ReadFromUDP reads a UDP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUDP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *UDPConn) ReadFromUDP(b []byte) (n int, addr *UDPAddr, err error) {
	if !c.ok() || c.fd.data == nil {
		return 0, nil, syscall.EINVAL
	}
	buf := make([]byte, udpHeaderSize+len(b))
	m, err := c.fd.data.Read(buf)
	if err != nil {
		return
	}
	if m < udpHeaderSize {
		return 0, nil, errors.New("short read reading UDP header")
	}
	buf = buf[:m]

	h, buf := unmarshalUDPHeader(buf)
	n = copy(b, buf)
	return n, &UDPAddr{IP: h.raddr, Port: int(h.rport)}, nil
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UDPConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	return c.ReadFromUDP(b)
}

// ReadMsgUDP reads a packet from c, copying the payload into b and
// the associated out-of-band data into oob.  It returns the number
// of bytes copied into b, the number of bytes copied into oob, the
// flags that were set on the packet and the source address of the
// packet.
func (c *UDPConn) ReadMsgUDP(b, oob []byte) (n, oobn, flags int, addr *UDPAddr, err error) {
	return 0, 0, 0, nil, syscall.EPLAN9
}

// WriteToUDP writes a UDP packet to addr via c, copying the payload
// from b.
//
// WriteToUDP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *UDPConn) WriteToUDP(b []byte, addr *UDPAddr) (int, error) {
	if !c.ok() || c.fd.data == nil {
		return 0, syscall.EINVAL
	}
	if addr == nil {
		return 0, &OpError{Op: "write", Net: c.fd.dir, Addr: nil, Err: errMissingAddress}
	}
	h := new(udpHeader)
	h.raddr = addr.IP.To16()
	h.laddr = c.fd.laddr.(*UDPAddr).IP.To16()
	h.ifcaddr = IPv6zero // ignored (receive only)
	h.rport = uint16(addr.Port)
	h.lport = uint16(c.fd.laddr.(*UDPAddr).Port)

	buf := make([]byte, udpHeaderSize+len(b))
	i := copy(buf, h.Bytes())
	copy(buf[i:], b)
	return c.fd.data.Write(buf)
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UDPConn) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*UDPAddr)
	if !ok {
		return 0, &OpError{"write", c.fd.dir, addr, syscall.EINVAL}
	}
	return c.WriteToUDP(b, a)
}

// WriteMsgUDP writes a packet to addr via c, copying the payload from
// b and the associated out-of-band data from oob.  It returns the
// number of payload and out-of-band bytes written.
func (c *UDPConn) WriteMsgUDP(b, oob []byte, addr *UDPAddr) (n, oobn int, err error) {
	return 0, 0, syscall.EPLAN9
}

// DialUDP connects to the remote address raddr on the network net,
// which must be "udp", "udp4", or "udp6".  If laddr is not nil, it is
// used as the local address for the connection.
func DialUDP(net string, laddr, raddr *UDPAddr) (*UDPConn, error) {
	return dialUDP(net, laddr, raddr, noDeadline)
}

func dialUDP(net string, laddr, raddr *UDPAddr, deadline time.Time) (*UDPConn, error) {
	if !deadline.IsZero() {
		panic("net.dialUDP: deadline not implemented on Plan 9")
	}
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}
	fd, err := dialPlan9(net, laddr, raddr)
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
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

// ListenUDP listens for incoming UDP packets addressed to the local
// address laddr.  Net must be "udp", "udp4", or "udp6".  If laddr has
// a port of 0, ListenUDP will choose an available port.
// The LocalAddr method of the returned UDPConn can be used to
// discover the port.  The returned connection's ReadFrom and WriteTo
// methods can be used to receive and send UDP packets with per-packet
// addressing.
func ListenUDP(net string, laddr *UDPAddr) (*UDPConn, error) {
	switch net {
	case "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		laddr = &UDPAddr{}
	}
	l, err := listenPlan9(net, laddr)
	if err != nil {
		return nil, err
	}
	_, err = l.ctl.WriteString("headers")
	if err != nil {
		return nil, err
	}
	l.data, err = os.OpenFile(l.dir+"/data", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	fd, err := l.netFD()
	return newUDPConn(fd), err
}

// ListenMulticastUDP listens for incoming multicast UDP packets
// addressed to the group address gaddr on ifi, which specifies the
// interface to join.  ListenMulticastUDP uses default multicast
// interface if ifi is nil.
func ListenMulticastUDP(net string, ifi *Interface, gaddr *UDPAddr) (*UDPConn, error) {
	return nil, syscall.EPLAN9
}
