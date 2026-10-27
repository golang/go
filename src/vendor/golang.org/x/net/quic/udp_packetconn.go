// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"net"
	"net/netip"
)

// netPacketConn is a packetConn implementation wrapping a net.PacketConn.
//
// This is mostly useful for tests, since PacketConn doesn't provide access to
// important features such as identifying the local address packets were received on.
type netPacketConn struct {
	c         net.PacketConn
	localAddr netip.AddrPort
}

func newNetPacketConn(pc net.PacketConn) (*netPacketConn, error) {
	addr, err := addrPortFromAddr(pc.LocalAddr())
	if err != nil {
		return nil, err
	}
	return &netPacketConn{
		c:         pc,
		localAddr: addr,
	}, nil
}

func (c *netPacketConn) Close() error {
	return c.c.Close()
}

func (c *netPacketConn) LocalAddr() netip.AddrPort {
	return c.localAddr
}

func (c *netPacketConn) LocalAddrFor(remote netip.AddrPort) (netip.AddrPort, error) {
	if c.localAddr.Addr().IsUnspecified() {
		// If this is the unspec address, we have no way to pick a better one.
		return netip.AddrPort{}, nil
	}
	return c.localAddr, nil
}

func (c *netPacketConn) Read(f func(*datagram)) error {
	for {
		dgram := newDatagram()
		n, peerAddr, err := c.c.ReadFrom(dgram.b)
		if err != nil {
			return err
		}
		dgram.path.peer, err = addrPortFromAddr(peerAddr)
		if err != nil {
			continue
		}
		dgram.b = dgram.b[:n]
		f(dgram)
	}
}

func (c *netPacketConn) Write(dgram datagram) error {
	_, err := c.c.WriteTo(dgram.b, net.UDPAddrFromAddrPort(dgram.path.peer))
	return err
}

func addrPortFromAddr(addr net.Addr) (netip.AddrPort, error) {
	switch a := addr.(type) {
	case *net.UDPAddr:
		return a.AddrPort(), nil
	}
	return netip.ParseAddrPort(addr.String())
}
