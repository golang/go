// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification for Linux

package net

import (
	"os"
	"syscall"
	"unsafe"
)

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otheriwse it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, os.Error) {
	var (
		ift  []Interface
		tab  []byte
		msgs []syscall.NetlinkMessage
		e    int
	)

	tab, e = syscall.NetlinkRIB(syscall.RTM_GETLINK, syscall.AF_UNSPEC)
	if e != 0 {
		return nil, os.NewSyscallError("netlink rib", e)
	}

	msgs, e = syscall.ParseNetlinkMessage(tab)
	if e != 0 {
		return nil, os.NewSyscallError("netlink message", e)
	}

	for _, m := range msgs {
		switch m.Header.Type {
		case syscall.NLMSG_DONE:
			goto done
		case syscall.RTM_NEWLINK:
			ifim := (*syscall.IfInfomsg)(unsafe.Pointer(&m.Data[0]))
			if ifindex == 0 || ifindex == int(ifim.Index) {
				attrs, e := syscall.ParseNetlinkRouteAttr(&m)
				if e != 0 {
					return nil, os.NewSyscallError("netlink routeattr", e)
				}
				ifi := newLink(attrs, ifim)
				ift = append(ift, ifi)
			}
		}
	}

done:
	return ift, nil
}

func newLink(attrs []syscall.NetlinkRouteAttr, ifim *syscall.IfInfomsg) Interface {
	ifi := Interface{Index: int(ifim.Index), rawFlags: int(ifim.Flags)}
	for _, a := range attrs {
		switch a.Attr.Type {
		case syscall.IFLA_ADDRESS:
			var nonzero bool
			for _, b := range a.Value {
				if b != 0 {
					nonzero = true
				}
			}
			if nonzero {
				ifi.HardwareAddr = a.Value[:]
			}
		case syscall.IFLA_IFNAME:
			ifi.Name = string(a.Value[:])
		case syscall.IFLA_MTU:
			ifi.MTU = int(uint32(a.Value[3])<<24 | uint32(a.Value[2])<<16 | uint32(a.Value[1])<<8 | uint32(a.Value[0]))
		}
	}
	return ifi
}

// If the ifindex is zero, interfaceAddrTable returns addresses
// for all network interfaces.  Otherwise it returns addresses
// for a specific interface.
func interfaceAddrTable(ifindex int) ([]Addr, os.Error) {
	var (
		ifat4 []Addr
		ifat6 []Addr
		tab   []byte
		msgs4 []syscall.NetlinkMessage
		msgs6 []syscall.NetlinkMessage
		e     int
		err   os.Error
	)

	tab, e = syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_INET)
	if e != 0 {
		return nil, os.NewSyscallError("netlink rib", e)
	}
	msgs4, e = syscall.ParseNetlinkMessage(tab)
	if e != 0 {
		return nil, os.NewSyscallError("netlink message", e)
	}
	ifat4, err = addrTable(msgs4, ifindex)
	if err != nil {
		return nil, err
	}

	tab, e = syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_INET6)
	if e != 0 {
		return nil, os.NewSyscallError("netlink rib", e)
	}
	msgs6, e = syscall.ParseNetlinkMessage(tab)
	if e != 0 {
		return nil, os.NewSyscallError("netlink message", e)
	}
	ifat6, err = addrTable(msgs6, ifindex)
	if err != nil {
		return nil, err
	}

	return append(ifat4, ifat6...), nil
}

func addrTable(msgs []syscall.NetlinkMessage, ifindex int) ([]Addr, os.Error) {
	var ifat []Addr

	for _, m := range msgs {
		switch m.Header.Type {
		case syscall.NLMSG_DONE:
			goto done
		case syscall.RTM_NEWADDR:
			ifam := (*syscall.IfAddrmsg)(unsafe.Pointer(&m.Data[0]))
			if ifindex == 0 || ifindex == int(ifam.Index) {
				attrs, e := syscall.ParseNetlinkRouteAttr(&m)
				if e != 0 {
					return nil, os.NewSyscallError("netlink routeattr", e)
				}
				ifat = append(ifat, newAddr(attrs, int(ifam.Family))...)
			}
		}
	}

done:
	return ifat, nil
}

func newAddr(attrs []syscall.NetlinkRouteAttr, family int) []Addr {
	var ifat []Addr

	for _, a := range attrs {
		switch a.Attr.Type {
		case syscall.IFA_ADDRESS:
			ifa := IPAddr{}
			switch family {
			case syscall.AF_INET:
				ifa.IP = IPv4(a.Value[0], a.Value[1], a.Value[2], a.Value[3])
			case syscall.AF_INET6:
				ifa.IP = make(IP, IPv6len)
				copy(ifa.IP, a.Value[:])
			}
			ifat = append(ifat, ifa.toAddr())
		}
	}

	return ifat
}
