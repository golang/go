// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import "unsafe"

func (typ RIBType) parseable() bool { return true }

// A RouteMetrics represents route metrics.
type RouteMetrics struct {
	PathMTU int // path maximum transmission unit
}

// SysType implements the SysType method of Sys interface.
func (rmx *RouteMetrics) SysType() SysType { return SysMetrics }

// Sys implements the Sys method of Message interface.
func (m *RouteMessage) Sys() []Sys {
	return []Sys{
		&RouteMetrics{
			PathMTU: int(nativeEndian.Uint64(m.raw[m.extOff+8 : m.extOff+16])),
		},
	}
}

// A InterfaceMetrics represents interface metrics.
type InterfaceMetrics struct {
	Type int // interface type
	MTU  int // maximum transmission unit
}

// SysType implements the SysType method of Sys interface.
func (imx *InterfaceMetrics) SysType() SysType { return SysMetrics }

// Sys implements the Sys method of Message interface.
func (m *InterfaceMessage) Sys() []Sys {
	return []Sys{
		&InterfaceMetrics{
			Type: int(m.raw[m.extOff]),
			MTU:  int(nativeEndian.Uint32(m.raw[m.extOff+8 : m.extOff+12])),
		},
	}
}

func probeRoutingStack() (int, map[int]parseFn) {
	var p uintptr
	rtm := &wireFormat{extOff: 40, bodyOff: sizeofRtMsghdrDragonFlyBSD4}
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrDragonFlyBSD4}
	ifam := &wireFormat{extOff: sizeofIfaMsghdrDragonFlyBSD4, bodyOff: sizeofIfaMsghdrDragonFlyBSD4}
	ifmam := &wireFormat{extOff: sizeofIfmaMsghdrDragonFlyBSD4, bodyOff: sizeofIfmaMsghdrDragonFlyBSD4}
	ifanm := &wireFormat{extOff: sizeofIfAnnouncemsghdrDragonFlyBSD4, bodyOff: sizeofIfAnnouncemsghdrDragonFlyBSD4}
	return int(unsafe.Sizeof(p)), map[int]parseFn{
		sysRTM_ADD:        rtm.parseRouteMessage,
		sysRTM_DELETE:     rtm.parseRouteMessage,
		sysRTM_CHANGE:     rtm.parseRouteMessage,
		sysRTM_GET:        rtm.parseRouteMessage,
		sysRTM_LOSING:     rtm.parseRouteMessage,
		sysRTM_REDIRECT:   rtm.parseRouteMessage,
		sysRTM_MISS:       rtm.parseRouteMessage,
		sysRTM_LOCK:       rtm.parseRouteMessage,
		sysRTM_RESOLVE:    rtm.parseRouteMessage,
		sysRTM_NEWADDR:    ifam.parseInterfaceAddrMessage,
		sysRTM_DELADDR:    ifam.parseInterfaceAddrMessage,
		sysRTM_IFINFO:     ifm.parseInterfaceMessage,
		sysRTM_NEWMADDR:   ifmam.parseInterfaceMulticastAddrMessage,
		sysRTM_DELMADDR:   ifmam.parseInterfaceMulticastAddrMessage,
		sysRTM_IFANNOUNCE: ifanm.parseInterfaceAnnounceMessage,
	}
}
