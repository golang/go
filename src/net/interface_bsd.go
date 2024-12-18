// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package net

import (
	"internal/routebsd"
	"syscall"
)

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces. Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	msgs, err := interfaceMessages(ifindex)
	if err != nil {
		return nil, err
	}
	n := len(msgs)
	if ifindex != 0 {
		n = 1
	}
	ift := make([]Interface, n)
	n = 0
	for _, m := range msgs {
		switch m := m.(type) {
		case *routebsd.InterfaceMessage:
			if ifindex != 0 && ifindex != m.Index {
				continue
			}
			ift[n].Index = m.Index
			ift[n].Name = m.Name
			ift[n].Flags = linkFlags(m.Flags)
			if sa, ok := m.Addrs[syscall.RTAX_IFP].(*routebsd.LinkAddr); ok && len(sa.Addr) > 0 {
				ift[n].HardwareAddr = make([]byte, len(sa.Addr))
				copy(ift[n].HardwareAddr, sa.Addr)
			}
			for _, sys := range m.Sys() {
				if imx, ok := sys.(*routebsd.InterfaceMetrics); ok {
					ift[n].MTU = imx.MTU
					break
				}
			}
			n++
			if ifindex == m.Index {
				return ift[:n], nil
			}
		}
	}
	return ift[:n], nil
}

func linkFlags(rawFlags int) Flags {
	var f Flags
	if rawFlags&syscall.IFF_UP != 0 {
		f |= FlagUp
	}
	if rawFlags&syscall.IFF_RUNNING != 0 {
		f |= FlagRunning
	}
	if rawFlags&syscall.IFF_BROADCAST != 0 {
		f |= FlagBroadcast
	}
	if rawFlags&syscall.IFF_LOOPBACK != 0 {
		f |= FlagLoopback
	}
	if rawFlags&syscall.IFF_POINTOPOINT != 0 {
		f |= FlagPointToPoint
	}
	if rawFlags&syscall.IFF_MULTICAST != 0 {
		f |= FlagMulticast
	}
	return f
}

// If the ifi is nil, interfaceAddrTable returns addresses for all
// network interfaces. Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	index := 0
	if ifi != nil {
		index = ifi.Index
	}
	msgs, err := interfaceMessages(index)
	if err != nil {
		return nil, err
	}
	ifat := make([]Addr, 0, len(msgs))
	for _, m := range msgs {
		switch m := m.(type) {
		case *routebsd.InterfaceAddrMessage:
			if index != 0 && index != m.Index {
				continue
			}
			var mask IPMask
			switch sa := m.Addrs[syscall.RTAX_NETMASK].(type) {
			case *routebsd.Inet4Addr:
				mask = IPv4Mask(sa.IP[0], sa.IP[1], sa.IP[2], sa.IP[3])
			case *routebsd.Inet6Addr:
				mask = make(IPMask, IPv6len)
				copy(mask, sa.IP[:])
			}
			var ip IP
			switch sa := m.Addrs[syscall.RTAX_IFA].(type) {
			case *routebsd.Inet4Addr:
				ip = IPv4(sa.IP[0], sa.IP[1], sa.IP[2], sa.IP[3])
			case *routebsd.Inet6Addr:
				ip = make(IP, IPv6len)
				copy(ip, sa.IP[:])
			}
			if ip != nil && mask != nil { // NetBSD may contain routebsd.LinkAddr
				ifat = append(ifat, &IPNet{IP: ip, Mask: mask})
			}
		}
	}
	return ifat, nil
}
