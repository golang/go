// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Routing sockets and messages for FreeBSD

package syscall

import "unsafe"

func (any *anyMessage) toRoutingMessage(b []byte) RoutingMessage {
	switch any.Type {
	case RTM_ADD, RTM_DELETE, RTM_CHANGE, RTM_GET, RTM_LOSING, RTM_REDIRECT, RTM_MISS, RTM_LOCK, RTM_RESOLVE:
		p := (*RouteMessage)(unsafe.Pointer(any))
		return &RouteMessage{Header: p.Header, Data: b[SizeofRtMsghdr:any.Msglen]}
	case RTM_IFINFO:
		p := (*InterfaceMessage)(unsafe.Pointer(any))
		return &InterfaceMessage{Header: p.Header, Data: b[SizeofIfMsghdr:any.Msglen]}
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
// network interface arrival and depature information.
type InterfaceAnnounceMessage struct {
	Header IfAnnounceMsghdr
}

func (m *InterfaceAnnounceMessage) sockaddr() (sas []Sockaddr) { return nil }

// InterfaceMulticastAddrMessage represents a routing message
// containing network interface address entries.
type InterfaceMulticastAddrMessage struct {
	Header IfmaMsghdr
	Data   []byte
}

const rtaIfmaMask = RTA_GATEWAY | RTA_IFP | RTA_IFA

func (m *InterfaceMulticastAddrMessage) sockaddr() (sas []Sockaddr) {
	if m.Header.Addrs&rtaIfmaMask == 0 {
		return nil
	}
	b := m.Data[:]
	for i := uint(0); i < RTAX_MAX; i++ {
		if m.Header.Addrs&rtaIfmaMask&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch i {
		case RTAX_IFA:
			sa, e := anyToSockaddr((*RawSockaddrAny)(unsafe.Pointer(rsa)))
			if e != nil {
				return nil
			}
			sas = append(sas, sa)
		case RTAX_GATEWAY, RTAX_IFP:
			// nothing to do
		}
		b = b[rsaAlignOf(int(rsa.Len)):]
	}
	return sas
}
