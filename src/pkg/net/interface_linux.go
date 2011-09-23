// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification for Linux

package net

import (
	"fmt"
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
	ifi := Interface{Index: int(ifim.Index), Flags: linkFlags(ifim.Flags)}
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
			ifi.Name = string(a.Value[:len(a.Value)-1])
		case syscall.IFLA_MTU:
			ifi.MTU = int(uint32(a.Value[3])<<24 | uint32(a.Value[2])<<16 | uint32(a.Value[1])<<8 | uint32(a.Value[0]))
		}
	}
	return ifi
}

func linkFlags(rawFlags uint32) Flags {
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

// If the ifindex is zero, interfaceAddrTable returns addresses
// for all network interfaces.  Otherwise it returns addresses
// for a specific interface.
func interfaceAddrTable(ifindex int) ([]Addr, os.Error) {
	var (
		tab  []byte
		e    int
		err  os.Error
		ifat []Addr
		msgs []syscall.NetlinkMessage
	)

	tab, e = syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_UNSPEC)
	if e != 0 {
		return nil, os.NewSyscallError("netlink rib", e)
	}

	msgs, e = syscall.ParseNetlinkMessage(tab)
	if e != 0 {
		return nil, os.NewSyscallError("netlink message", e)
	}

	ifat, err = addrTable(msgs, ifindex)
	if err != nil {
		return nil, err
	}

	return ifat, nil
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
			switch family {
			case syscall.AF_INET:
				ifa := &IPAddr{IP: IPv4(a.Value[0], a.Value[1], a.Value[2], a.Value[3])}
				ifat = append(ifat, ifa.toAddr())
			case syscall.AF_INET6:
				ifa := &IPAddr{IP: make(IP, IPv6len)}
				copy(ifa.IP, a.Value[:])
				ifat = append(ifat, ifa.toAddr())
			}
		}
	}

	return ifat
}

// If the ifindex is zero, interfaceMulticastAddrTable returns
// addresses for all network interfaces.  Otherwise it returns
// addresses for a specific interface.
func interfaceMulticastAddrTable(ifindex int) ([]Addr, os.Error) {
	var (
		ifi *Interface
		err os.Error
	)

	if ifindex > 0 {
		ifi, err = InterfaceByIndex(ifindex)
		if err != nil {
			return nil, err
		}
	}

	ifmat4 := parseProcNetIGMP(ifi)
	ifmat6 := parseProcNetIGMP6(ifi)

	return append(ifmat4, ifmat6...), nil
}

func parseProcNetIGMP(ifi *Interface) []Addr {
	var (
		ifmat []Addr
		name  string
	)

	fd, err := open("/proc/net/igmp")
	if err != nil {
		return nil
	}
	defer fd.close()

	fd.readLine() // skip first line
	b := make([]byte, IPv4len)
	for l, ok := fd.readLine(); ok; l, ok = fd.readLine() {
		f := getFields(l)
		switch len(f) {
		case 4:
			if ifi == nil || name == ifi.Name {
				fmt.Sscanf(f[0], "%08x", &b)
				ifma := IPAddr{IP: IPv4(b[3], b[2], b[1], b[0])}
				ifmat = append(ifmat, ifma.toAddr())
			}
		case 5:
			name = f[1]
		}
	}

	return ifmat
}

func parseProcNetIGMP6(ifi *Interface) []Addr {
	var ifmat []Addr

	fd, err := open("/proc/net/igmp6")
	if err != nil {
		return nil
	}
	defer fd.close()

	b := make([]byte, IPv6len)
	for l, ok := fd.readLine(); ok; l, ok = fd.readLine() {
		f := getFields(l)
		if ifi == nil || f[1] == ifi.Name {
			fmt.Sscanf(f[2], "%32x", &b)
			ifma := IPAddr{IP: IP{b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]}}
			ifmat = append(ifmat, ifma.toAddr())

		}
	}

	return ifmat
}
