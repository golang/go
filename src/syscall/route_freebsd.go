// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"internal/stringslite"
	"unsafe"
)

func init() {
	machine, _ := Sysctl("hw.machine")
	if machine == "i386" {
		arches, _ := Sysctl("hw.supported_archs")
		amd64 := "amd64"
		if stringslite.Index(arches, amd64) >= 0 {
			machine = amd64
		}
	}
	freebsdConfArch = machine
}

func (any *anyMessage) toRoutingMessage(b []byte) RoutingMessage {
	switch any.Type {
	case RTM_ADD, RTM_DELETE, RTM_CHANGE, RTM_GET, RTM_LOSING, RTM_REDIRECT, RTM_MISS, RTM_LOCK, RTM_RESOLVE:
		return any.parseRouteMessage(b)
	case RTM_IFINFO:
		return any.parseInterfaceMessage(b)
	case RTM_IFANNOUNCE:
		p := (*InterfaceAnnounceMessage)(unsafe.Pointer(any))
		return &InterfaceAnnounceMessage{Header: p.Header}
	case RTM_NEWADDR, RTM_DELADDR:
		p := (*InterfaceAddrMessage)(unsafe.Pointer(any))
		return &InterfaceAddrMessage{Header: p.Header, Data: b[SizeofIfaMsghdr:any.Msglen]}
	case RTM_NEWMADDR, RTM_DELMADDR:
		p := (*InterfaceMulticastAddrMessage)(unsafe.Pointer(any))
		return &InterfaceMulticastAddrMessage{Header: p.Header, Data: b[SizeofIfmaMsghdr:any.Msglen]}
	}
	return nil
}

// InterfaceAnnounceMessage represents a routing message containing
// network interface arrival and departure information.
//
// Deprecated: Use golang.org/x/net/route instead.
type InterfaceAnnounceMessage struct {
	Header IfAnnounceMsghdr
}

func (m *InterfaceAnnounceMessage) sockaddr() ([]Sockaddr, error) { return nil, nil }

// InterfaceMulticastAddrMessage represents a routing message
// containing network interface address entries.
//
// Deprecated: Use golang.org/x/net/route instead.
type InterfaceMulticastAddrMessage struct {
	Header IfmaMsghdr
	Data   []byte
}

func (m *InterfaceMulticastAddrMessage) sockaddr() ([]Sockaddr, error) {
	var sas [RTAX_MAX]Sockaddr
	b := m.Data[:]
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
		default:
			sa, l, err := parseLinkLayerAddr(b)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[l:]
		}
	}
	return sas[:], nil
}
