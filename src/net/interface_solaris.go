// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"

	"golang_org/x/net/lif"
)

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces. Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	lls, err := lif.Links(syscall.AF_UNSPEC, "")
	if err != nil {
		return nil, err
	}
	var ift []Interface
	for _, ll := range lls {
		if ifindex != 0 && ifindex != ll.Index {
			continue
		}
		ifi := Interface{Index: ll.Index, MTU: ll.MTU, Name: ll.Name, Flags: linkFlags(ll.Flags)}
		if len(ll.Addr) > 0 {
			ifi.HardwareAddr = HardwareAddr(ll.Addr)
		}
		ift = append(ift, ifi)
	}
	return ift, nil
}

const (
	sysIFF_UP          = 0x1
	sysIFF_BROADCAST   = 0x2
	sysIFF_DEBUG       = 0x4
	sysIFF_LOOPBACK    = 0x8
	sysIFF_POINTOPOINT = 0x10
	sysIFF_NOTRAILERS  = 0x20
	sysIFF_RUNNING     = 0x40
	sysIFF_NOARP       = 0x80
	sysIFF_PROMISC     = 0x100
	sysIFF_ALLMULTI    = 0x200
	sysIFF_INTELLIGENT = 0x400
	sysIFF_MULTICAST   = 0x800
	sysIFF_MULTI_BCAST = 0x1000
	sysIFF_UNNUMBERED  = 0x2000
	sysIFF_PRIVATE     = 0x8000
)

func linkFlags(rawFlags int) Flags {
	var f Flags
	if rawFlags&sysIFF_UP != 0 {
		f |= FlagUp
	}
	if rawFlags&sysIFF_BROADCAST != 0 {
		f |= FlagBroadcast
	}
	if rawFlags&sysIFF_LOOPBACK != 0 {
		f |= FlagLoopback
	}
	if rawFlags&sysIFF_POINTOPOINT != 0 {
		f |= FlagPointToPoint
	}
	if rawFlags&sysIFF_MULTICAST != 0 {
		f |= FlagMulticast
	}
	return f
}

// If the ifi is nil, interfaceAddrTable returns addresses for all
// network interfaces. Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	var name string
	if ifi != nil {
		name = ifi.Name
	}
	as, err := lif.Addrs(syscall.AF_UNSPEC, name)
	if err != nil {
		return nil, err
	}
	var ifat []Addr
	for _, a := range as {
		var ip IP
		var mask IPMask
		switch a := a.(type) {
		case *lif.Inet4Addr:
			ip = IPv4(a.IP[0], a.IP[1], a.IP[2], a.IP[3])
			mask = CIDRMask(a.PrefixLen, 8*IPv4len)
		case *lif.Inet6Addr:
			ip = make(IP, IPv6len)
			copy(ip, a.IP[:])
			mask = CIDRMask(a.PrefixLen, 8*IPv6len)
		}
		ifat = append(ifat, &IPNet{IP: ip, Mask: mask})
	}
	return ifat, nil
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	return nil, nil
}
