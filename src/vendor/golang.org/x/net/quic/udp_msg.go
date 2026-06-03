// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !quicbasicnet && (darwin || linux)

package quic

import (
	"encoding/binary"
	"net"
	"net/netip"
	"sync"
	"syscall"
	"unsafe"
)

// Network interface for platforms using sendmsg/recvmsg with cmsgs.

type netUDPConn struct {
	c         *net.UDPConn
	localAddr netip.AddrPort
}

func newNetUDPConn(uc *net.UDPConn) (*netUDPConn, error) {
	a, _ := uc.LocalAddr().(*net.UDPAddr)
	localAddr := a.AddrPort()
	if localAddr.Addr().IsUnspecified() {
		// If the conn is not bound to a specified (non-wildcard) address,
		// then set localAddr.Addr to an invalid netip.Addr.
		// This better conveys that this is not an address we should be using,
		// and is a bit more efficient to test against.
		localAddr = netip.AddrPortFrom(netip.Addr{}, localAddr.Port())
	}

	sc, err := uc.SyscallConn()
	if err != nil {
		return nil, err
	}
	sc.Control(func(fd uintptr) {
		// Ask for ECN info and (when we aren't bound to a fixed local address)
		// destination info.
		//
		// If any of these calls fail, we won't get the requested information.
		// That's fine, we'll gracefully handle the lack.
		syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IP, ip_recvtos, 1)
		syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IPV6, syscall.IPV6_RECVTCLASS, 1)
		if !localAddr.IsValid() {
			syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IP, syscall.IP_PKTINFO, 1)
			syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IPV6, ipv6_recvpktinfo, 1)
		}
	})

	return &netUDPConn{
		c:         uc,
		localAddr: localAddr,
	}, nil
}

func (c *netUDPConn) Close() error { return c.c.Close() }

func (c *netUDPConn) LocalAddr() netip.AddrPort {
	a, _ := c.c.LocalAddr().(*net.UDPAddr)
	return a.AddrPort()
}

func (c *netUDPConn) Read(f func(*datagram)) {
	// We shouldn't ever see all of these messages at the same time,
	// but the total is small so just allocate enough space for everything we use.
	const (
		inPktinfoSize  = 12 // int + in_addr + in_addr
		in6PktinfoSize = 20 // in6_addr + int
		ipTOSSize      = 4
		ipv6TclassSize = 4
	)
	control := make([]byte, 0+
		syscall.CmsgSpace(inPktinfoSize)+
		syscall.CmsgSpace(in6PktinfoSize)+
		syscall.CmsgSpace(ipTOSSize)+
		syscall.CmsgSpace(ipv6TclassSize))

	for {
		d := newDatagram()
		n, controlLen, _, peerAddr, err := c.c.ReadMsgUDPAddrPort(d.b, control)
		if err != nil {
			return
		}
		if n == 0 {
			continue
		}
		d.localAddr = c.localAddr
		d.peerAddr = unmapAddrPort(peerAddr)
		d.b = d.b[:n]
		parseControl(d, control[:controlLen])
		f(d)
	}
}

var cmsgPool = sync.Pool{
	New: func() any {
		return new([]byte)
	},
}

func (c *netUDPConn) Write(dgram datagram) error {
	controlp := cmsgPool.Get().(*[]byte)
	control := *controlp
	defer func() {
		*controlp = control[:0]
		cmsgPool.Put(controlp)
	}()

	localIP := dgram.localAddr.Addr()
	if localIP.IsValid() {
		if localIP.Is4() {
			control = appendCmsgIPSourceAddrV4(control, localIP)
		} else {
			control = appendCmsgIPSourceAddrV6(control, localIP)
		}
	}
	if dgram.ecn != ecnNotECT {
		if dgram.peerAddr.Addr().Is4() {
			control = appendCmsgECNv4(control, dgram.ecn)
		} else {
			control = appendCmsgECNv6(control, dgram.ecn)
		}
	}

	_, _, err := c.c.WriteMsgUDPAddrPort(dgram.b, control, dgram.peerAddr)
	return err
}

func parseControl(d *datagram, control []byte) {
	msgs, err := syscall.ParseSocketControlMessage(control)
	if err != nil {
		return
	}
	for _, m := range msgs {
		switch m.Header.Level {
		case syscall.IPPROTO_IP:
			switch m.Header.Type {
			case syscall.IP_TOS, ip_recvtos:
				// (Linux sets the type to IP_TOS, Darwin to IP_RECVTOS,
				// just check for both.)
				if ecn, ok := parseIPTOS(m.Data); ok {
					d.ecn = ecn
				}
			case syscall.IP_PKTINFO:
				if a, ok := parseInPktinfo(m.Data); ok {
					d.localAddr = netip.AddrPortFrom(a, d.localAddr.Port())
				}
			}
		case syscall.IPPROTO_IPV6:
			switch m.Header.Type {
			case syscall.IPV6_TCLASS:
				// 32-bit integer containing the traffic class field.
				// The low two bits are the ECN field.
				if ecn, ok := parseIPv6TCLASS(m.Data); ok {
					d.ecn = ecn
				}
			case ipv6_pktinfo:
				if a, ok := parseIn6Pktinfo(m.Data); ok {
					d.localAddr = netip.AddrPortFrom(a, d.localAddr.Port())
				}
			}
		}
	}
}

// IPV6_TCLASS is specified by RFC 3542 as an int.

func parseIPv6TCLASS(b []byte) (ecnBits, bool) {
	if len(b) != 4 {
		return 0, false
	}
	return ecnBits(binary.NativeEndian.Uint32(b) & ecnMask), true
}

func appendCmsgECNv6(b []byte, ecn ecnBits) []byte {
	b, data := appendCmsg(b, syscall.IPPROTO_IPV6, syscall.IPV6_TCLASS, 4)
	binary.NativeEndian.PutUint32(data, uint32(ecn))
	return b
}

// struct in_pktinfo {
//   unsigned int   ipi_ifindex;  /* send/recv interface index */
//   struct in_addr ipi_spec_dst; /* Local address */
//   struct in_addr ipi_addr;     /* IP Header dst address */
// };

// parseInPktinfo returns the destination address from an IP_PKTINFO.
func parseInPktinfo(b []byte) (dst netip.Addr, ok bool) {
	if len(b) != 12 {
		return netip.Addr{}, false
	}
	return netip.AddrFrom4([4]byte(b[8:][:4])), true
}

// appendCmsgIPSourceAddrV4 appends an IP_PKTINFO setting the source address
// for an outbound datagram.
func appendCmsgIPSourceAddrV4(b []byte, src netip.Addr) []byte {
	// struct in_pktinfo {
	//   unsigned int   ipi_ifindex;  /* send/recv interface index */
	//   struct in_addr ipi_spec_dst; /* Local address */
	//   struct in_addr ipi_addr;     /* IP Header dst address */
	// };
	b, data := appendCmsg(b, syscall.IPPROTO_IP, syscall.IP_PKTINFO, 12)
	ip := src.As4()
	copy(data[4:], ip[:])
	return b
}

// struct in6_pktinfo {
//   struct in6_addr  ipi6_addr;    /* src/dst IPv6 address */
//   unsigned int     ipi6_ifindex; /* send/recv interface index */
// };

// parseIn6Pktinfo returns the destination address from an IPV6_PKTINFO.
func parseIn6Pktinfo(b []byte) (netip.Addr, bool) {
	if len(b) != 20 {
		return netip.Addr{}, false
	}
	return netip.AddrFrom16([16]byte(b[:16])).Unmap(), true
}

// appendCmsgIPSourceAddrV6 appends an IPV6_PKTINFO setting the source address
// for an outbound datagram.
func appendCmsgIPSourceAddrV6(b []byte, src netip.Addr) []byte {
	b, data := appendCmsg(b, syscall.IPPROTO_IPV6, ipv6_pktinfo, 20)
	ip := src.As16()
	copy(data[0:], ip[:])
	return b
}

// appendCmsg appends a cmsg with the given level, type, and size to b.
// It returns the new buffer, and the data section of the cmsg.
func appendCmsg(b []byte, level, typ int32, size int) (_, data []byte) {
	off := len(b)
	b = append(b, make([]byte, syscall.CmsgSpace(size))...)
	h := (*syscall.Cmsghdr)(unsafe.Pointer(&b[off]))
	h.Level = level
	h.Type = typ
	h.SetLen(syscall.CmsgLen(size))
	return b, b[off+syscall.CmsgSpace(0):][:size]
}
