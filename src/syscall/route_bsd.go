// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package syscall

import (
	"runtime"
	"unsafe"
)

var (
	freebsdConfArch       string // "machine $arch" line in kern.conftxt on freebsd
	minRoutingSockaddrLen = rsaAlignOf(0)
)

// Round the length of a raw sockaddr up to align it properly.
func rsaAlignOf(salen int) int {
	salign := sizeofPtr
	if darwin64Bit {
		// Darwin kernels require 32-bit aligned access to
		// routing facilities.
		salign = 4
	} else if netbsd32Bit {
		// NetBSD 6 and beyond kernels require 64-bit aligned
		// access to routing facilities.
		salign = 8
	} else if runtime.GOOS == "freebsd" {
		// In the case of kern.supported_archs="amd64 i386",
		// we need to know the underlying kernel's
		// architecture because the alignment for routing
		// facilities are set at the build time of the kernel.
		if freebsdConfArch == "amd64" {
			salign = 8
		}
	}
	if salen == 0 {
		return salign
	}
	return (salen + salign - 1) & ^(salign - 1)
}

// parseSockaddrLink parses b as a datalink socket address.
func parseSockaddrLink(b []byte) (*SockaddrDatalink, error) {
	if len(b) < 8 {
		return nil, EINVAL
	}
	sa, _, err := parseLinkLayerAddr(b[4:])
	if err != nil {
		return nil, err
	}
	rsa := (*RawSockaddrDatalink)(unsafe.Pointer(&b[0]))
	sa.Len = rsa.Len
	sa.Family = rsa.Family
	sa.Index = rsa.Index
	return sa, nil
}

// parseLinkLayerAddr parses b as a datalink socket address in
// conventional BSD kernel form.
func parseLinkLayerAddr(b []byte) (*SockaddrDatalink, int, error) {
	// The encoding looks like the following:
	// +----------------------------+
	// | Type             (1 octet) |
	// +----------------------------+
	// | Name length      (1 octet) |
	// +----------------------------+
	// | Address length   (1 octet) |
	// +----------------------------+
	// | Selector length  (1 octet) |
	// +----------------------------+
	// | Data            (variable) |
	// +----------------------------+
	type linkLayerAddr struct {
		Type byte
		Nlen byte
		Alen byte
		Slen byte
	}
	lla := (*linkLayerAddr)(unsafe.Pointer(&b[0]))
	l := 4 + int(lla.Nlen) + int(lla.Alen) + int(lla.Slen)
	if len(b) < l {
		return nil, 0, EINVAL
	}
	b = b[4:]
	sa := &SockaddrDatalink{Type: lla.Type, Nlen: lla.Nlen, Alen: lla.Alen, Slen: lla.Slen}
	for i := 0; len(sa.Data) > i && i < l-4; i++ {
		sa.Data[i] = int8(b[i])
	}
	return sa, rsaAlignOf(l), nil
}

// parseSockaddrInet parses b as an internet socket address.
func parseSockaddrInet(b []byte, family byte) (Sockaddr, error) {
	switch family {
	case AF_INET:
		if len(b) < SizeofSockaddrInet4 {
			return nil, EINVAL
		}
		rsa := (*RawSockaddrAny)(unsafe.Pointer(&b[0]))
		return anyToSockaddr(rsa)
	case AF_INET6:
		if len(b) < SizeofSockaddrInet6 {
			return nil, EINVAL
		}
		rsa := (*RawSockaddrAny)(unsafe.Pointer(&b[0]))
		return anyToSockaddr(rsa)
	default:
		return nil, EINVAL
	}
}

const (
	offsetofInet4 = int(unsafe.Offsetof(RawSockaddrInet4{}.Addr))
	offsetofInet6 = int(unsafe.Offsetof(RawSockaddrInet6{}.Addr))
)

// parseNetworkLayerAddr parses b as an internet socket address in
// conventional BSD kernel form.
func parseNetworkLayerAddr(b []byte, family byte) (Sockaddr, error) {
	// The encoding looks similar to the NLRI encoding.
	// +----------------------------+
	// | Length           (1 octet) |
	// +----------------------------+
	// | Address prefix  (variable) |
	// +----------------------------+
	//
	// The differences between the kernel form and the NLRI
	// encoding are:
	//
	// - The length field of the kernel form indicates the prefix
	//   length in bytes, not in bits
	//
	// - In the kernel form, zero value of the length field
	//   doesn't mean 0.0.0.0/0 or ::/0
	//
	// - The kernel form appends leading bytes to the prefix field
	//   to make the <length, prefix> tuple to be conformed with
	//   the routing message boundary
	l := int(rsaAlignOf(int(b[0])))
	if len(b) < l {
		return nil, EINVAL
	}
	// Don't reorder case expressions.
	// The case expressions for IPv6 must come first.
	switch {
	case b[0] == SizeofSockaddrInet6:
		sa := &SockaddrInet6{}
		copy(sa.Addr[:], b[offsetofInet6:])
		return sa, nil
	case family == AF_INET6:
		sa := &SockaddrInet6{}
		if l-1 < offsetofInet6 {
			copy(sa.Addr[:], b[1:l])
		} else {
			copy(sa.Addr[:], b[l-offsetofInet6:l])
		}
		return sa, nil
	case b[0] == SizeofSockaddrInet4:
		sa := &SockaddrInet4{}
		copy(sa.Addr[:], b[offsetofInet4:])
		return sa, nil
	default: // an old fashion, AF_UNSPEC or unknown means AF_INET
		sa := &SockaddrInet4{}
		if l-1 < offsetofInet4 {
			copy(sa.Addr[:], b[1:l])
		} else {
			copy(sa.Addr[:], b[l-offsetofInet4:l])
		}
		return sa, nil
	}
}

// RouteRIB returns routing information base, as known as RIB,
// which consists of network facility information, states and
// parameters.
//
// Deprecated: Use golang.org/x/net/route instead.
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
//
// Deprecated: Use golang.org/x/net/route instead.
type RoutingMessage interface {
	sockaddr() ([]Sockaddr, error)
}

const anyMessageLen = int(unsafe.Sizeof(anyMessage{}))

type anyMessage struct {
	Msglen  uint16
	Version uint8
	Type    uint8
}

// RouteMessage represents a routing message containing routing
// entries.
//
// Deprecated: Use golang.org/x/net/route instead.
type RouteMessage struct {
	Header RtMsghdr
	Data   []byte
}

func (m *RouteMessage) sockaddr() ([]Sockaddr, error) {
	var sas [RTAX_MAX]Sockaddr
	b := m.Data[:]
	family := uint8(AF_UNSPEC)
	for i := uint(0); i < RTAX_MAX && len(b) >= minRoutingSockaddrLen; i++ {
		if m.Header.Addrs&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch rsa.Family {
		case AF_LINK:
			sa, err := parseSockaddrLink(b)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
		case AF_INET, AF_INET6:
			sa, err := parseSockaddrInet(b, rsa.Family)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
			family = rsa.Family
		default:
			sa, err := parseNetworkLayerAddr(b, family)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(b[0])):]
		}
	}
	return sas[:], nil
}

// InterfaceMessage represents a routing message containing
// network interface entries.
//
// Deprecated: Use golang.org/x/net/route instead.
type InterfaceMessage struct {
	Header IfMsghdr
	Data   []byte
}

func (m *InterfaceMessage) sockaddr() ([]Sockaddr, error) {
	var sas [RTAX_MAX]Sockaddr
	if m.Header.Addrs&RTA_IFP == 0 {
		return nil, nil
	}
	sa, err := parseSockaddrLink(m.Data[:])
	if err != nil {
		return nil, err
	}
	sas[RTAX_IFP] = sa
	return sas[:], nil
}

// InterfaceAddrMessage represents a routing message containing
// network interface address entries.
//
// Deprecated: Use golang.org/x/net/route instead.
type InterfaceAddrMessage struct {
	Header IfaMsghdr
	Data   []byte
}

func (m *InterfaceAddrMessage) sockaddr() ([]Sockaddr, error) {
	var sas [RTAX_MAX]Sockaddr
	b := m.Data[:]
	family := uint8(AF_UNSPEC)
	for i := uint(0); i < RTAX_MAX && len(b) >= minRoutingSockaddrLen; i++ {
		if m.Header.Addrs&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch rsa.Family {
		case AF_LINK:
			sa, err := parseSockaddrLink(b)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
		case AF_INET, AF_INET6:
			sa, err := parseSockaddrInet(b, rsa.Family)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
			family = rsa.Family
		default:
			sa, err := parseNetworkLayerAddr(b, family)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(b[0])):]
		}
	}
	return sas[:], nil
}

// ParseRoutingMessage parses b as routing messages and returns the
// slice containing the [RoutingMessage] interfaces.
//
// Deprecated: Use golang.org/x/net/route instead.
func ParseRoutingMessage(b []byte) (msgs []RoutingMessage, err error) {
	nmsgs, nskips := 0, 0
	for len(b) >= anyMessageLen {
		nmsgs++
		any := (*anyMessage)(unsafe.Pointer(&b[0]))
		if any.Version != RTM_VERSION {
			b = b[any.Msglen:]
			continue
		}
		if m := any.toRoutingMessage(b); m == nil {
			nskips++
		} else {
			msgs = append(msgs, m)
		}
		b = b[any.Msglen:]
	}
	// We failed to parse any of the messages - version mismatch?
	if nmsgs != len(msgs)+nskips {
		return nil, EINVAL
	}
	return msgs, nil
}

// ParseRoutingSockaddr parses msg's payload as raw sockaddrs and
// returns the slice containing the [Sockaddr] interfaces.
//
// Deprecated: Use golang.org/x/net/route instead.
func ParseRoutingSockaddr(msg RoutingMessage) ([]Sockaddr, error) {
	sas, err := msg.sockaddr()
	if err != nil {
		return nil, err
	}
	return sas, nil
}
