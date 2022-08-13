// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (freebsd && 386) || (freebsd && arm)

package syscall

import "unsafe"

func (any *anyMessage) parseRouteMessage(b []byte) *RouteMessage {
	p := (*RouteMessage)(unsafe.Pointer(any))
	off := int(unsafe.Offsetof(p.Header.Rmx)) + SizeofRtMetrics
	if freebsdConfArch == "amd64" {
		off += SizeofRtMetrics // rt_metrics on amd64 is simply doubled
	}
	return &RouteMessage{Header: p.Header, Data: b[rsaAlignOf(off):any.Msglen]}
}

func (any *anyMessage) parseInterfaceMessage(b []byte) *InterfaceMessage {
	p := (*InterfaceMessage)(unsafe.Pointer(any))
	// FreeBSD 10 and beyond have a restructured mbuf
	// packet header view.
	// See https://svnweb.freebsd.org/base?view=revision&revision=254804.
	if supportsABI(1000000) {
		m := (*ifMsghdr)(unsafe.Pointer(any))
		p.Header.Data.Hwassist = uint32(m.Data.Hwassist)
		p.Header.Data.Epoch = m.Data.Epoch
		p.Header.Data.Lastchange = m.Data.Lastchange
		return &InterfaceMessage{Header: p.Header, Data: b[int(unsafe.Offsetof(p.Header.Data))+int(p.Header.Data.Datalen) : any.Msglen]}
	}
	return &InterfaceMessage{Header: p.Header, Data: b[int(unsafe.Offsetof(p.Header.Data))+int(p.Header.Data.Datalen) : any.Msglen]}
}
