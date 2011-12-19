// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Routing sockets and messages for NetBSD

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
	}
	return nil
}
