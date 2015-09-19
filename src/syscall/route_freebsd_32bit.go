// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd,386 freebsd,arm

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
