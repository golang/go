// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	tab, err := syscall.NetlinkRIB(syscall.RTM_GETLINK, syscall.AF_UNSPEC)
	if err != nil {
		return nil, os.NewSyscallError("netlinkrib", err)
	}
	msgs, err := syscall.ParseNetlinkMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("parsenetlinkmessage", err)
	}
	var ift []Interface
loop:
	for _, m := range msgs {
		switch m.Header.Type {
		case syscall.NLMSG_DONE:
			break loop
		case syscall.RTM_NEWLINK:
			ifim := (*syscall.IfInfomsg)(unsafe.Pointer(&m.Data[0]))
			if ifindex == 0 || ifindex == int(ifim.Index) {
				attrs, err := syscall.ParseNetlinkRouteAttr(&m)
				if err != nil {
					return nil, os.NewSyscallError("parsenetlinkrouteattr", err)
				}
				ift = append(ift, *newLink(ifim, attrs))
				if ifindex == int(ifim.Index) {
					break loop
				}
			}
		}
	}
	return ift, nil
}

const (
	// See linux/if_arp.h.
	// Note that Linux doesn't support IPv4 over IPv6 tunneling.
	sysARPHardwareIPv4IPv4 = 768 // IPv4 over IPv4 tunneling
	sysARPHardwareIPv6IPv6 = 769 // IPv6 over IPv6 tunneling
	sysARPHardwareIPv6IPv4 = 776 // IPv6 over IPv4 tunneling
	sysARPHardwareGREIPv4  = 778 // any over GRE over IPv4 tunneling
	sysARPHardwareGREIPv6  = 823 // any over GRE over IPv6 tunneling
)

func newLink(ifim *syscall.IfInfomsg, attrs []syscall.NetlinkRouteAttr) *Interface {
	ifi := &Interface{Index: int(ifim.Index), Flags: linkFlags(ifim.Flags)}
	for _, a := range attrs {
		switch a.Attr.Type {
		case syscall.IFLA_ADDRESS:
			// We never return any /32 or /128 IP address
			// prefix on any IP tunnel interface as the
			// hardware address.
			switch len(a.Value) {
			case IPv4len:
				switch ifim.Type {
				case sysARPHardwareIPv4IPv4, sysARPHardwareGREIPv4, sysARPHardwareIPv6IPv4:
					continue
				}
			case IPv6len:
				switch ifim.Type {
				case sysARPHardwareIPv6IPv6, sysARPHardwareGREIPv6:
					continue
				}
			}
			var nonzero bool
			for _, b := range a.Value {
				if b != 0 {
					nonzero = true
					break
				}
			}
			if nonzero {
				ifi.HardwareAddr = a.Value[:]
			}
		case syscall.IFLA_IFNAME:
			ifi.Name = string(a.Value[:len(a.Value)-1])
		case syscall.IFLA_MTU:
			ifi.MTU = int(*(*uint32)(unsafe.Pointer(&a.Value[:4][0])))
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

// If the ifi is nil, interfaceAddrTable returns addresses for all
// network interfaces.  Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	tab, err := syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_UNSPEC)
	if err != nil {
		return nil, os.NewSyscallError("netlinkrib", err)
	}
	msgs, err := syscall.ParseNetlinkMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("parsenetlinkmessage", err)
	}
	var ift []Interface
	if ifi == nil {
		var err error
		ift, err = interfaceTable(0)
		if err != nil {
			return nil, err
		}
	}
	ifat, err := addrTable(ift, ifi, msgs)
	if err != nil {
		return nil, err
	}
	return ifat, nil
}

func addrTable(ift []Interface, ifi *Interface, msgs []syscall.NetlinkMessage) ([]Addr, error) {
	var ifat []Addr
loop:
	for _, m := range msgs {
		switch m.Header.Type {
		case syscall.NLMSG_DONE:
			break loop
		case syscall.RTM_NEWADDR:
			ifam := (*syscall.IfAddrmsg)(unsafe.Pointer(&m.Data[0]))
			if len(ift) != 0 || ifi.Index == int(ifam.Index) {
				if len(ift) != 0 {
					var err error
					ifi, err = interfaceByIndex(ift, int(ifam.Index))
					if err != nil {
						return nil, err
					}
				}
				attrs, err := syscall.ParseNetlinkRouteAttr(&m)
				if err != nil {
					return nil, os.NewSyscallError("parsenetlinkrouteattr", err)
				}
				ifa := newAddr(ifi, ifam, attrs)
				if ifa != nil {
					ifat = append(ifat, ifa)
				}
			}
		}
	}
	return ifat, nil
}

func newAddr(ifi *Interface, ifam *syscall.IfAddrmsg, attrs []syscall.NetlinkRouteAttr) Addr {
	var ipPointToPoint bool
	// Seems like we need to make sure whether the IP interface
	// stack consists of IP point-to-point numbered or unnumbered
	// addressing.
	for _, a := range attrs {
		if a.Attr.Type == syscall.IFA_LOCAL {
			ipPointToPoint = true
			break
		}
	}
	for _, a := range attrs {
		if ipPointToPoint && a.Attr.Type == syscall.IFA_ADDRESS {
			continue
		}
		switch ifam.Family {
		case syscall.AF_INET:
			return &IPNet{IP: IPv4(a.Value[0], a.Value[1], a.Value[2], a.Value[3]), Mask: CIDRMask(int(ifam.Prefixlen), 8*IPv4len)}
		case syscall.AF_INET6:
			ifa := &IPNet{IP: make(IP, IPv6len), Mask: CIDRMask(int(ifam.Prefixlen), 8*IPv6len)}
			copy(ifa.IP, a.Value[:])
			return ifa
		}
	}
	return nil
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	ifmat4 := parseProcNetIGMP("/proc/net/igmp", ifi)
	ifmat6 := parseProcNetIGMP6("/proc/net/igmp6", ifi)
	return append(ifmat4, ifmat6...), nil
}

func parseProcNetIGMP(path string, ifi *Interface) []Addr {
	fd, err := open(path)
	if err != nil {
		return nil
	}
	defer fd.close()
	var (
		ifmat []Addr
		name  string
	)
	fd.readLine() // skip first line
	b := make([]byte, IPv4len)
	for l, ok := fd.readLine(); ok; l, ok = fd.readLine() {
		f := splitAtBytes(l, " :\r\t\n")
		if len(f) < 4 {
			continue
		}
		switch {
		case l[0] != ' ' && l[0] != '\t': // new interface line
			name = f[1]
		case len(f[0]) == 8:
			if ifi == nil || name == ifi.Name {
				// The Linux kernel puts the IP
				// address in /proc/net/igmp in native
				// endianness.
				for i := 0; i+1 < len(f[0]); i += 2 {
					b[i/2], _ = xtoi2(f[0][i:i+2], 0)
				}
				i := *(*uint32)(unsafe.Pointer(&b[:4][0]))
				ifma := &IPAddr{IP: IPv4(byte(i>>24), byte(i>>16), byte(i>>8), byte(i))}
				ifmat = append(ifmat, ifma)
			}
		}
	}
	return ifmat
}

func parseProcNetIGMP6(path string, ifi *Interface) []Addr {
	fd, err := open(path)
	if err != nil {
		return nil
	}
	defer fd.close()
	var ifmat []Addr
	b := make([]byte, IPv6len)
	for l, ok := fd.readLine(); ok; l, ok = fd.readLine() {
		f := splitAtBytes(l, " \r\t\n")
		if len(f) < 6 {
			continue
		}
		if ifi == nil || f[1] == ifi.Name {
			for i := 0; i+1 < len(f[2]); i += 2 {
				b[i/2], _ = xtoi2(f[2][i:i+2], 0)
			}
			ifma := &IPAddr{IP: IP{b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]}}
			ifmat = append(ifmat, ifma)
		}
	}
	return ifmat
}
