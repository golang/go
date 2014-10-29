// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

// Routing sockets and messages

package syscall

import "unsafe"

// Round the length of a raw sockaddr up to align it properly.
func rsaAlignOf(salen int) int {
	salign := sizeofPtr
	// NOTE: It seems like 64-bit Darwin kernel still requires
	// 32-bit aligned access to BSD subsystem. Also NetBSD 6
	// kernel and beyond require 64-bit aligned access to routing
	// facilities.
	if darwin64Bit {
		salign = 4
	} else if netbsd32Bit {
		salign = 8
	}
	if salen == 0 {
		return salign
	}
	return (salen + salign - 1) & ^(salign - 1)
}

// RouteRIB returns routing information base, as known as RIB,
// which consists of network facility information, states and
// parameters.
func RouteRIB(facility, param int) ([]byte, error) {
	mib := []_C_int{CTL_NET, AF_ROUTE, 0, 0, _C_int(facility), _C_int(param)}
	// Find size.
	n := uintptr(0)
	if err := sysctl(mib, nil, &n, nil, 0); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	tab := make([]byte, n)
	if err := sysctl(mib, &tab[0], &n, nil, 0); err != nil {
		return nil, err
	}
	return tab[:n], nil
}

// RoutingMessage represents a routing message.
type RoutingMessage interface {
	sockaddr() []Sockaddr
}

const anyMessageLen = int(unsafe.Sizeof(anyMessage{}))

type anyMessage struct {
	Msglen  uint16
	Version uint8
	Type    uint8
}

// RouteMessage represents a routing message containing routing
// entries.
type RouteMessage struct {
	Header RtMsghdr
	Data   []byte
}

const rtaRtMask = RTA_DST | RTA_GATEWAY | RTA_NETMASK | RTA_GENMASK

func (m *RouteMessage) sockaddr() []Sockaddr {
	var (
		af  int
		sas [4]Sockaddr
	)
	b := m.Data[:]
	for i := uint(0); i < RTAX_MAX; i++ {
		if m.Header.Addrs&rtaRtMask&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch i {
		case RTAX_DST, RTAX_GATEWAY:
			sa, err := anyToSockaddr((*RawSockaddrAny)(unsafe.Pointer(rsa)))
			if err != nil {
				return nil
			}
			if i == RTAX_DST {
				af = int(rsa.Family)
			}
			sas[i] = sa
		case RTAX_NETMASK, RTAX_GENMASK:
			switch af {
			case AF_INET:
				rsa4 := (*RawSockaddrInet4)(unsafe.Pointer(&b[0]))
				sa := new(SockaddrInet4)
				for j := 0; rsa4.Len > 0 && j < int(rsa4.Len)-int(unsafe.Offsetof(rsa4.Addr)); j++ {
					sa.Addr[j] = rsa4.Addr[j]
				}
				sas[i] = sa
			case AF_INET6:
				rsa6 := (*RawSockaddrInet6)(unsafe.Pointer(&b[0]))
				sa := new(SockaddrInet6)
				for j := 0; rsa6.Len > 0 && j < int(rsa6.Len)-int(unsafe.Offsetof(rsa6.Addr)); j++ {
					sa.Addr[j] = rsa6.Addr[j]
				}
				sas[i] = sa
			}
		}
		b = b[rsaAlignOf(int(rsa.Len)):]
	}
	return sas[:]
}

// InterfaceMessage represents a routing message containing
// network interface entries.
type InterfaceMessage struct {
	Header IfMsghdr
	Data   []byte
}

func (m *InterfaceMessage) sockaddr() (sas []Sockaddr) {
	if m.Header.Addrs&RTA_IFP == 0 {
		return nil
	}
	sa, err := anyToSockaddr((*RawSockaddrAny)(unsafe.Pointer(&m.Data[0])))
	if err != nil {
		return nil
	}
	return append(sas, sa)
}

// InterfaceAddrMessage represents a routing message containing
// network interface address entries.
type InterfaceAddrMessage struct {
	Header IfaMsghdr
	Data   []byte
}

const rtaIfaMask = RTA_IFA | RTA_NETMASK | RTA_BRD

func (m *InterfaceAddrMessage) sockaddr() (sas []Sockaddr) {
	if m.Header.Addrs&rtaIfaMask == 0 {
		return nil
	}
	b := m.Data[:]
	// We still see AF_UNSPEC in socket addresses on some
	// platforms. To identify each address family correctly, we
	// will use the address family of RTAX_NETMASK as a preferred
	// one on the 32-bit NetBSD kernel, also use the length of
	// RTAX_NETMASK socket address on the FreeBSD kernel.
	preferredFamily := uint8(AF_UNSPEC)
	for i := uint(0); i < RTAX_MAX; i++ {
		if m.Header.Addrs&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch i {
		case RTAX_IFA:
			if rsa.Family == AF_UNSPEC {
				rsa.Family = preferredFamily
			}
			sa, err := anyToSockaddr((*RawSockaddrAny)(unsafe.Pointer(rsa)))
			if err != nil {
				return nil
			}
			sas = append(sas, sa)
		case RTAX_NETMASK:
			switch rsa.Family {
			case AF_UNSPEC:
				switch rsa.Len {
				case SizeofSockaddrInet4:
					rsa.Family = AF_INET
				case SizeofSockaddrInet6:
					rsa.Family = AF_INET6
				default:
					rsa.Family = AF_INET // an old fashion, AF_UNSPEC means AF_INET
				}
			case AF_INET, AF_INET6:
				preferredFamily = rsa.Family
			default:
				return nil
			}
			sa, err := anyToSockaddr((*RawSockaddrAny)(unsafe.Pointer(rsa)))
			if err != nil {
				return nil
			}
			sas = append(sas, sa)
		case RTAX_BRD:
			// nothing to do
		}
		b = b[rsaAlignOf(int(rsa.Len)):]
	}
	return sas
}

// ParseRoutingMessage parses b as routing messages and returns the
// slice containing the RoutingMessage interfaces.
func ParseRoutingMessage(b []byte) (msgs []RoutingMessage, err error) {
	msgCount := 0
	for len(b) >= anyMessageLen {
		msgCount++
		any := (*anyMessage)(unsafe.Pointer(&b[0]))
		if any.Version != RTM_VERSION {
			b = b[any.Msglen:]
			continue
		}
		msgs = append(msgs, any.toRoutingMessage(b))
		b = b[any.Msglen:]
	}
	// We failed to parse any of the messages - version mismatch?
	if msgCount > 0 && len(msgs) == 0 {
		return nil, EINVAL
	}
	return msgs, nil
}

// ParseRoutingMessage parses msg's payload as raw sockaddrs and
// returns the slice containing the Sockaddr interfaces.
func ParseRoutingSockaddr(msg RoutingMessage) (sas []Sockaddr, err error) {
	return append(sas, msg.sockaddr()...), nil
}
