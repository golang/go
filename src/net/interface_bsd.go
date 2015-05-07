// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package net

import (
	"os"
	"syscall"
	"unsafe"
)

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	tab, err := syscall.RouteRIB(syscall.NET_RT_IFLIST, ifindex)
	if err != nil {
		return nil, os.NewSyscallError("routerib", err)
	}
	msgs, err := syscall.ParseRoutingMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("parseroutingmessage", err)
	}
	return parseInterfaceTable(ifindex, msgs)
}

func parseInterfaceTable(ifindex int, msgs []syscall.RoutingMessage) ([]Interface, error) {
	var ift []Interface
loop:
	for _, m := range msgs {
		switch m := m.(type) {
		case *syscall.InterfaceMessage:
			if ifindex == 0 || ifindex == int(m.Header.Index) {
				ifi, err := newLink(m)
				if err != nil {
					return nil, err
				}
				ift = append(ift, *ifi)
				if ifindex == int(m.Header.Index) {
					break loop
				}
			}
		}
	}
	return ift, nil
}

func newLink(m *syscall.InterfaceMessage) (*Interface, error) {
	sas, err := syscall.ParseRoutingSockaddr(m)
	if err != nil {
		return nil, os.NewSyscallError("parseroutingsockaddr", err)
	}
	ifi := &Interface{Index: int(m.Header.Index), Flags: linkFlags(m.Header.Flags)}
	sa, _ := sas[syscall.RTAX_IFP].(*syscall.SockaddrDatalink)
	if sa != nil {
		// NOTE: SockaddrDatalink.Data is minimum work area,
		// can be larger.
		m.Data = m.Data[unsafe.Offsetof(sa.Data):]
		var name [syscall.IFNAMSIZ]byte
		for i := 0; i < int(sa.Nlen); i++ {
			name[i] = byte(m.Data[i])
		}
		ifi.Name = string(name[:sa.Nlen])
		ifi.MTU = int(m.Header.Data.Mtu)
		addr := make([]byte, sa.Alen)
		for i := 0; i < int(sa.Alen); i++ {
			addr[i] = byte(m.Data[int(sa.Nlen)+i])
		}
		ifi.HardwareAddr = addr[:sa.Alen]
	}
	return ifi, nil
}

func linkFlags(rawFlags int32) Flags {
	var f Flags
	if rawFlags&syscall.IFF_UP != 0 {
		f |= FlagUp
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
// network interfaces.  Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	index := 0
	if ifi != nil {
		index = ifi.Index
	}
	tab, err := syscall.RouteRIB(syscall.NET_RT_IFLIST, index)
	if err != nil {
		return nil, os.NewSyscallError("routerib", err)
	}
	msgs, err := syscall.ParseRoutingMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("parseroutingmessage", err)
	}
	var ift []Interface
	if index == 0 {
		ift, err = parseInterfaceTable(index, msgs)
		if err != nil {
			return nil, err
		}
	}
	var ifat []Addr
	for _, m := range msgs {
		switch m := m.(type) {
		case *syscall.InterfaceAddrMessage:
			if index == 0 || index == int(m.Header.Index) {
				if index == 0 {
					var err error
					ifi, err = interfaceByIndex(ift, int(m.Header.Index))
					if err != nil {
						return nil, err
					}
				}
				ifa, err := newAddr(ifi, m)
				if err != nil {
					return nil, err
				}
				if ifa != nil {
					ifat = append(ifat, ifa)
				}
			}
		}
	}
	return ifat, nil
}

func newAddr(ifi *Interface, m *syscall.InterfaceAddrMessage) (*IPNet, error) {
	sas, err := syscall.ParseRoutingSockaddr(m)
	if err != nil {
		return nil, os.NewSyscallError("parseroutingsockaddr", err)
	}
	ifa := &IPNet{}
	switch sa := sas[syscall.RTAX_NETMASK].(type) {
	case *syscall.SockaddrInet4:
		ifa.Mask = IPv4Mask(sa.Addr[0], sa.Addr[1], sa.Addr[2], sa.Addr[3])
	case *syscall.SockaddrInet6:
		ifa.Mask = make(IPMask, IPv6len)
		copy(ifa.Mask, sa.Addr[:])
	}
	switch sa := sas[syscall.RTAX_IFA].(type) {
	case *syscall.SockaddrInet4:
		ifa.IP = IPv4(sa.Addr[0], sa.Addr[1], sa.Addr[2], sa.Addr[3])
	case *syscall.SockaddrInet6:
		ifa.IP = make(IP, IPv6len)
		copy(ifa.IP, sa.Addr[:])
		// NOTE: KAME based IPv6 protcol stack usually embeds
		// the interface index in the interface-local or
		// link-local address as the kernel-internal form.
		if ifa.IP.IsLinkLocalUnicast() {
			ifa.IP[2], ifa.IP[3] = 0, 0
		}
	}
	if ifa.IP == nil || ifa.Mask == nil {
		return nil, nil // Sockaddrs contain syscall.SockaddrDatalink on NetBSD
	}
	return ifa, nil
}
