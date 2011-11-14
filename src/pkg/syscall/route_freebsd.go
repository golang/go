// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Routing sockets and messages for FreeBSD

package syscall

import (
	"unsafe"
)

func (any *anyMessage) toRoutingMessage(buf []byte) RoutingMessage {
	switch any.Type {
	case RTM_ADD, RTM_DELETE, RTM_CHANGE, RTM_GET, RTM_LOSING, RTM_REDIRECT, RTM_MISS, RTM_LOCK, RTM_RESOLVE:
		p := (*RouteMessage)(unsafe.Pointer(any))
		rtm := &RouteMessage{}
		rtm.Header = p.Header
		rtm.Data = buf[SizeofRtMsghdr:any.Msglen]
		return rtm
	case RTM_IFINFO:
		p := (*InterfaceMessage)(unsafe.Pointer(any))
		ifm := &InterfaceMessage{}
		ifm.Header = p.Header
		ifm.Data = buf[SizeofIfMsghdr:any.Msglen]
		return ifm
	case RTM_NEWADDR, RTM_DELADDR:
		p := (*InterfaceAddrMessage)(unsafe.Pointer(any))
		ifam := &InterfaceAddrMessage{}
		ifam.Header = p.Header
		ifam.Data = buf[SizeofIfaMsghdr:any.Msglen]
		return ifam
	case RTM_NEWMADDR, RTM_DELMADDR:
		p := (*InterfaceMulticastAddrMessage)(unsafe.Pointer(any))
		ifmam := &InterfaceMulticastAddrMessage{}
		ifmam.Header = p.Header
		ifmam.Data = buf[SizeofIfmaMsghdr:any.Msglen]
		return ifmam
	}
	return nil
}

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

	buf := m.Data[:]
	for i := uint(0); i < RTAX_MAX; i++ {
		if m.Header.Addrs&rtaIfmaMask&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&buf[0]))
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
		buf = buf[rsaAlignOf(int(rsa.Len)):]
	}

	return sas
}
