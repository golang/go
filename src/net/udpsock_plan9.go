// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"os"
	"syscall"
)

func (c *UDPConn) readFrom(b []byte) (n int, addr *UDPAddr, err error) {
	buf := make([]byte, udpHeaderSize+len(b))
	m, err := c.fd.Read(buf)
	if err != nil {
		return 0, nil, err
	}
	if m < udpHeaderSize {
		return 0, nil, errors.New("short read reading UDP header")
	}
	buf = buf[:m]

	h, buf := unmarshalUDPHeader(buf)
	n = copy(b, buf)
	return n, &UDPAddr{IP: h.raddr, Port: int(h.rport)}, nil
}

func (c *UDPConn) readMsg(b, oob []byte) (n, oobn, flags int, addr *UDPAddr, err error) {
	return 0, 0, 0, nil, syscall.EPLAN9
}

func (c *UDPConn) writeTo(b []byte, addr *UDPAddr) (int, error) {
	if addr == nil {
		return 0, errMissingAddress
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
	if _, err := c.fd.Write(buf); err != nil {
		return 0, err
	}
	return len(b), nil
}

func (c *UDPConn) writeMsg(b, oob []byte, addr *UDPAddr) (n, oobn int, err error) {
	return 0, 0, syscall.EPLAN9
}

func dialUDP(ctx context.Context, net string, laddr, raddr *UDPAddr) (*UDPConn, error) {
	fd, err := dialPlan9(ctx, net, laddr, raddr)
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

func listenUDP(ctx context.Context, network string, laddr *UDPAddr) (*UDPConn, error) {
	l, err := listenPlan9(ctx, network, laddr)
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

func listenMulticastUDP(ctx context.Context, network string, ifi *Interface, gaddr *UDPAddr) (*UDPConn, error) {
	l, err := listenPlan9(ctx, network, gaddr)
	if err != nil {
		return nil, err
	}
	_, err = l.ctl.WriteString("headers")
	if err != nil {
		return nil, err
	}
	var addrs []Addr
	if ifi != nil {
		addrs, err = ifi.Addrs()
		if err != nil {
			return nil, err
		}
	} else {
		addrs, err = InterfaceAddrs()
		if err != nil {
			return nil, err
		}
	}
	for _, addr := range addrs {
		if ipnet, ok := addr.(*IPNet); ok {
			_, err = l.ctl.WriteString("addmulti " + ipnet.IP.String() + " " + gaddr.IP.String())
			if err != nil {
				return nil, err
			}
		}
	}
	l.data, err = os.OpenFile(l.dir+"/data", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	fd, err := l.netFD()
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}
