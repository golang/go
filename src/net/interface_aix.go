// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"internal/syscall/unix"
	"syscall"
	"unsafe"
)

type rawSockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [120]byte
}

type ifreq struct {
	Name [16]uint8
	Ifru [16]byte
}

const _KINFO_RT_IFLIST = (0x1 << 8) | 3 | (1 << 30)

const _RTAX_NETMASK = 2
const _RTAX_IFA = 5
const _RTAX_MAX = 8

func getIfList() ([]byte, error) {
	needed, err := syscall.Getkerninfo(_KINFO_RT_IFLIST, 0, 0, 0)
	if err != nil {
		return nil, err
	}
	tab := make([]byte, needed)
	_, err = syscall.Getkerninfo(_KINFO_RT_IFLIST, uintptr(unsafe.Pointer(&tab[0])), uintptr(unsafe.Pointer(&needed)), 0)
	if err != nil {
		return nil, err
	}
	return tab[:needed], nil
}

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces. Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	tab, err := getIfList()
	if err != nil {
		return nil, err
	}

	sock, err := sysSocket(syscall.AF_INET, syscall.SOCK_DGRAM, 0)
	if err != nil {
		return nil, err
	}
	defer poll.CloseFunc(sock)

	var ift []Interface
	for len(tab) > 0 {
		ifm := (*syscall.IfMsgHdr)(unsafe.Pointer(&tab[0]))
		if ifm.Msglen == 0 {
			break
		}
		if ifm.Type == syscall.RTM_IFINFO {
			if ifindex == 0 || ifindex == int(ifm.Index) {
				sdl := (*rawSockaddrDatalink)(unsafe.Pointer(&tab[syscall.SizeofIfMsghdr]))

				ifi := &Interface{Index: int(ifm.Index), Flags: linkFlags(ifm.Flags)}
				ifi.Name = string(sdl.Data[:sdl.Nlen])
				ifi.HardwareAddr = sdl.Data[sdl.Nlen : sdl.Nlen+sdl.Alen]

				// Retrieve MTU
				ifr := &ifreq{}
				copy(ifr.Name[:], ifi.Name)
				err = unix.Ioctl(sock, syscall.SIOCGIFMTU, unsafe.Pointer(ifr))
				if err != nil {
					return nil, err
				}
				ifi.MTU = int(ifr.Ifru[0])<<24 | int(ifr.Ifru[1])<<16 | int(ifr.Ifru[2])<<8 | int(ifr.Ifru[3])

				ift = append(ift, *ifi)
				if ifindex == int(ifm.Index) {
					break
				}
			}
		}
		tab = tab[ifm.Msglen:]
	}

	return ift, nil
}

func linkFlags(rawFlags int32) Flags {
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
	tab, err := getIfList()
	if err != nil {
		return nil, err
	}

	var ifat []Addr
	for len(tab) > 0 {
		ifm := (*syscall.IfMsgHdr)(unsafe.Pointer(&tab[0]))
		if ifm.Msglen == 0 {
			break
		}
		if ifm.Type == syscall.RTM_NEWADDR {
			if ifi == nil || ifi.Index == int(ifm.Index) {
				mask := ifm.Addrs
				off := uint(syscall.SizeofIfMsghdr)

				var iprsa, nmrsa *syscall.RawSockaddr
				for i := uint(0); i < _RTAX_MAX; i++ {
					if mask&(1<<i) == 0 {
						continue
					}
					rsa := (*syscall.RawSockaddr)(unsafe.Pointer(&tab[off]))
					if i == _RTAX_NETMASK {
						nmrsa = rsa
					}
					if i == _RTAX_IFA {
						iprsa = rsa
					}
					off += (uint(rsa.Len) + 3) &^ 3
				}
				if iprsa != nil && nmrsa != nil {
					var mask IPMask
					var ip IP

					switch iprsa.Family {
					case syscall.AF_INET:
						ipsa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(iprsa))
						nmsa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(nmrsa))
						ip = IPv4(ipsa.Addr[0], ipsa.Addr[1], ipsa.Addr[2], ipsa.Addr[3])
						mask = IPv4Mask(nmsa.Addr[0], nmsa.Addr[1], nmsa.Addr[2], nmsa.Addr[3])
					case syscall.AF_INET6:
						ipsa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(iprsa))
						nmsa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(nmrsa))
						ip = make(IP, IPv6len)
						copy(ip, ipsa.Addr[:])
						mask = make(IPMask, IPv6len)
						copy(mask, nmsa.Addr[:])
					}
					ifa := &IPNet{IP: ip, Mask: mask}
					ifat = append(ifat, ifa)
				}
			}
		}
		tab = tab[ifm.Msglen:]
	}

	return ifat, nil
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	return nil, nil
}
