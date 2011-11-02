// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification for Windows

package net

import (
	"os"
	"syscall"
	"unsafe"
)

func bytePtrToString(p *uint8) string {
	a := (*[10000]uint8)(unsafe.Pointer(p))
	i := 0
	for a[i] != 0 {
		i++
	}
	return string(a[:i])
}

func getAdapterList() (*syscall.IpAdapterInfo, error) {
	b := make([]byte, 1000)
	l := uint32(len(b))
	a := (*syscall.IpAdapterInfo)(unsafe.Pointer(&b[0]))
	e := syscall.GetAdaptersInfo(a, &l)
	if e == syscall.ERROR_BUFFER_OVERFLOW {
		b = make([]byte, l)
		a = (*syscall.IpAdapterInfo)(unsafe.Pointer(&b[0]))
		e = syscall.GetAdaptersInfo(a, &l)
	}
	if e != 0 {
		return nil, os.NewSyscallError("GetAdaptersInfo", e)
	}
	return a, nil
}

func getInterfaceList() ([]syscall.InterfaceInfo, error) {
	s, e := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM, syscall.IPPROTO_UDP)
	if e != 0 {
		return nil, os.NewSyscallError("Socket", e)
	}
	defer syscall.Closesocket(s)

	ii := [20]syscall.InterfaceInfo{}
	ret := uint32(0)
	size := uint32(unsafe.Sizeof(ii))
	e = syscall.WSAIoctl(s, syscall.SIO_GET_INTERFACE_LIST, nil, 0, (*byte)(unsafe.Pointer(&ii[0])), size, &ret, nil, 0)
	if e != 0 {
		return nil, os.NewSyscallError("WSAIoctl", e)
	}
	c := ret / uint32(unsafe.Sizeof(ii[0]))
	return ii[:c-1], nil
}

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otheriwse it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	ai, e := getAdapterList()
	if e != nil {
		return nil, e
	}

	ii, e := getInterfaceList()
	if e != nil {
		return nil, e
	}

	var ift []Interface
	for ; ai != nil; ai = ai.Next {
		index := ai.Index
		if ifindex == 0 || ifindex == int(index) {
			var flags Flags

			row := syscall.MibIfRow{Index: index}
			e := syscall.GetIfEntry(&row)
			if e != 0 {
				return nil, os.NewSyscallError("GetIfEntry", e)
			}

			for _, ii := range ii {
				ip := (*syscall.RawSockaddrInet4)(unsafe.Pointer(&ii.Address)).Addr
				ipv4 := IPv4(ip[0], ip[1], ip[2], ip[3])
				ipl := &ai.IpAddressList
				for ipl != nil {
					ips := bytePtrToString(&ipl.IpAddress.String[0])
					if ipv4.Equal(parseIPv4(ips)) {
						break
					}
					ipl = ipl.Next
				}
				if ipl == nil {
					continue
				}
				if ii.Flags&syscall.IFF_UP != 0 {
					flags |= FlagUp
				}
				if ii.Flags&syscall.IFF_LOOPBACK != 0 {
					flags |= FlagLoopback
				}
				if ii.Flags&syscall.IFF_BROADCAST != 0 {
					flags |= FlagBroadcast
				}
				if ii.Flags&syscall.IFF_POINTTOPOINT != 0 {
					flags |= FlagPointToPoint
				}
				if ii.Flags&syscall.IFF_MULTICAST != 0 {
					flags |= FlagMulticast
				}
			}

			name := bytePtrToString(&ai.AdapterName[0])

			ifi := Interface{
				Index:        int(index),
				MTU:          int(row.Mtu),
				Name:         name,
				HardwareAddr: HardwareAddr(row.PhysAddr[:row.PhysAddrLen]),
				Flags:        flags}
			ift = append(ift, ifi)
		}
	}
	return ift, nil
}

// If the ifindex is zero, interfaceAddrTable returns addresses
// for all network interfaces.  Otherwise it returns addresses
// for a specific interface.
func interfaceAddrTable(ifindex int) ([]Addr, error) {
	ai, e := getAdapterList()
	if e != nil {
		return nil, e
	}

	var ifat []Addr
	for ; ai != nil; ai = ai.Next {
		index := ai.Index
		if ifindex == 0 || ifindex == int(index) {
			ipl := &ai.IpAddressList
			for ; ipl != nil; ipl = ipl.Next {
				ifa := IPAddr{}
				ifa.IP = parseIPv4(bytePtrToString(&ipl.IpAddress.String[0]))
				ifat = append(ifat, ifa.toAddr())
			}
		}
	}
	return ifat, nil
}

// If the ifindex is zero, interfaceMulticastAddrTable returns
// addresses for all network interfaces.  Otherwise it returns
// addresses for a specific interface.
func interfaceMulticastAddrTable(ifindex int) ([]Addr, error) {
	return nil, nil
}
