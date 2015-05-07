// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"os"
	"syscall"
	"unsafe"
)

func getAdapters() (*windows.IpAdapterAddresses, error) {
	block := uint32(unsafe.Sizeof(windows.IpAdapterAddresses{}))

	// pre-allocate a 15KB working buffer pointed to by the AdapterAddresses
	// parameter.
	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa365915(v=vs.85).aspx
	size := uint32(15000)

	var addrs []windows.IpAdapterAddresses
	for {
		addrs = make([]windows.IpAdapterAddresses, size/block+1)
		err := windows.GetAdaptersAddresses(syscall.AF_UNSPEC, windows.GAA_FLAG_INCLUDE_PREFIX, 0, &addrs[0], &size)
		if err == nil {
			break
		}
		if err.(syscall.Errno) != syscall.ERROR_BUFFER_OVERFLOW {
			return nil, os.NewSyscallError("getadaptersaddresses", err)
		}
	}
	return &addrs[0], nil
}

func getInterfaceInfos() ([]syscall.InterfaceInfo, error) {
	s, err := sysSocket(syscall.AF_INET, syscall.SOCK_DGRAM, syscall.IPPROTO_UDP)
	if err != nil {
		return nil, err
	}
	defer closeFunc(s)

	iia := [20]syscall.InterfaceInfo{}
	ret := uint32(0)
	size := uint32(unsafe.Sizeof(iia))
	err = syscall.WSAIoctl(s, syscall.SIO_GET_INTERFACE_LIST, nil, 0, (*byte)(unsafe.Pointer(&iia[0])), size, &ret, nil, 0)
	if err != nil {
		return nil, os.NewSyscallError("wsaioctl", err)
	}
	iilen := ret / uint32(unsafe.Sizeof(iia[0]))
	return iia[:iilen-1], nil
}

func bytesEqualIP(a []byte, b []int8) bool {
	for i := 0; i < len(a); i++ {
		if a[i] != byte(b[i]) {
			return false
		}
	}
	return true
}

func findInterfaceInfo(iis []syscall.InterfaceInfo, paddr *windows.IpAdapterAddresses) *syscall.InterfaceInfo {
	for _, ii := range iis {
		iaddr := (*syscall.RawSockaddr)(unsafe.Pointer(&ii.Address))
		puni := paddr.FirstUnicastAddress
		for ; puni != nil; puni = puni.Next {
			if iaddr.Family == puni.Address.Sockaddr.Addr.Family {
				switch iaddr.Family {
				case syscall.AF_INET:
					a := (*syscall.RawSockaddrInet4)(unsafe.Pointer(&ii.Address)).Addr
					if bytesEqualIP(a[:], puni.Address.Sockaddr.Addr.Data[2:]) {
						return &ii
					}
				case syscall.AF_INET6:
					a := (*syscall.RawSockaddrInet6)(unsafe.Pointer(&ii.Address)).Addr
					if bytesEqualIP(a[:], puni.Address.Sockaddr.Addr.Data[2:]) {
						return &ii
					}
				default:
					continue
				}
			}
		}
	}
	return nil
}

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	paddr, err := getAdapters()
	if err != nil {
		return nil, err
	}

	iis, err := getInterfaceInfos()
	if err != nil {
		return nil, err
	}

	var ift []Interface
	for ; paddr != nil; paddr = paddr.Next {
		index := paddr.IfIndex
		if paddr.Ipv6IfIndex != 0 {
			index = paddr.Ipv6IfIndex
		}
		if ifindex == 0 || ifindex == int(index) {
			ii := findInterfaceInfo(iis, paddr)
			if ii == nil {
				continue
			}
			var flags Flags
			if paddr.Flags&windows.IfOperStatusUp != 0 {
				flags |= FlagUp
			}
			if paddr.IfType&windows.IF_TYPE_SOFTWARE_LOOPBACK != 0 {
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
			ifi := Interface{
				Index:        int(index),
				MTU:          int(paddr.Mtu),
				Name:         syscall.UTF16ToString((*(*[10000]uint16)(unsafe.Pointer(paddr.FriendlyName)))[:]),
				HardwareAddr: HardwareAddr(paddr.PhysicalAddress[:]),
				Flags:        flags,
			}
			ift = append(ift, ifi)
			if ifindex == int(ifi.Index) {
				break
			}
		}
	}
	return ift, nil
}

// If the ifi is nil, interfaceAddrTable returns addresses for all
// network interfaces.  Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	paddr, err := getAdapters()
	if err != nil {
		return nil, err
	}

	var ifat []Addr
	for ; paddr != nil; paddr = paddr.Next {
		index := paddr.IfIndex
		if paddr.Ipv6IfIndex != 0 {
			index = paddr.Ipv6IfIndex
		}
		if ifi == nil || ifi.Index == int(index) {
			puni := paddr.FirstUnicastAddress
			for ; puni != nil; puni = puni.Next {
				if sa, err := puni.Address.Sockaddr.Sockaddr(); err == nil {
					switch sav := sa.(type) {
					case *syscall.SockaddrInet4:
						ifa := &IPNet{IP: make(IP, IPv4len), Mask: CIDRMask(int(puni.Address.SockaddrLength), 8*IPv4len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					case *syscall.SockaddrInet6:
						ifa := &IPNet{IP: make(IP, IPv6len), Mask: CIDRMask(int(puni.Address.SockaddrLength), 8*IPv6len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					}
				}
			}
			pany := paddr.FirstAnycastAddress
			for ; pany != nil; pany = pany.Next {
				if sa, err := pany.Address.Sockaddr.Sockaddr(); err == nil {
					switch sav := sa.(type) {
					case *syscall.SockaddrInet4:
						ifa := &IPNet{IP: make(IP, IPv4len), Mask: CIDRMask(int(pany.Address.SockaddrLength), 8*IPv4len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					case *syscall.SockaddrInet6:
						ifa := &IPNet{IP: make(IP, IPv6len), Mask: CIDRMask(int(pany.Address.SockaddrLength), 8*IPv6len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					}
				}
			}
		}
	}

	return ifat, nil
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	paddr, err := getAdapters()
	if err != nil {
		return nil, err
	}

	var ifat []Addr
	for ; paddr != nil; paddr = paddr.Next {
		index := paddr.IfIndex
		if paddr.Ipv6IfIndex != 0 {
			index = paddr.Ipv6IfIndex
		}
		if ifi == nil || ifi.Index == int(index) {
			pmul := paddr.FirstMulticastAddress
			for ; pmul != nil; pmul = pmul.Next {
				if sa, err := pmul.Address.Sockaddr.Sockaddr(); err == nil {
					switch sav := sa.(type) {
					case *syscall.SockaddrInet4:
						ifa := &IPAddr{IP: make(IP, IPv4len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					case *syscall.SockaddrInet6:
						ifa := &IPAddr{IP: make(IP, IPv6len)}
						copy(ifa.IP, sav.Addr[:])
						ifat = append(ifat, ifa)
					}
				}
			}
		}
	}

	return ifat, nil
}
