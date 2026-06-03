// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build quicbasicnet || !(darwin || linux)

package quic

import (
	"net"
	"net/netip"
)

// Lowest common denominator network interface: Basic net.UDPConn, no cmsgs.
// We will not be able to send or receive ECN bits,
// and we will not know what our local address is.
//
// The quicbasicnet build tag allows selecting this interface on any platform.

// See udp.go.
const (
	udpECNSupport              = false
	udpInvalidLocalAddrIsError = false
)

type netUDPConn struct {
	c *net.UDPConn
}

func newNetUDPConn(uc *net.UDPConn) (*netUDPConn, error) {
	return &netUDPConn{
		c: uc,
	}, nil
}

func (c *netUDPConn) Close() error { return c.c.Close() }

func (c *netUDPConn) LocalAddr() netip.AddrPort {
	a, _ := c.c.LocalAddr().(*net.UDPAddr)
	return a.AddrPort()
}

func (c *netUDPConn) Read(f func(*datagram)) {
	for {
		dgram := newDatagram()
		n, peerAddr, err := c.c.ReadFromUDPAddrPort(dgram.b)
		if err != nil {
			return
		}
		if n == 0 {
			continue
		}
		dgram.peerAddr = unmapAddrPort(peerAddr)
		dgram.b = dgram.b[:n]
		f(dgram)
	}
}

func (c *netUDPConn) Write(dgram datagram) error {
	_, err := c.c.WriteToUDPAddrPort(dgram.b, dgram.peerAddr)
	return err
}
