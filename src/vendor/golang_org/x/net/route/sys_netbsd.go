// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

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
	rtm := &wireFormat{extOff: 40, bodyOff: sizeofRtMsghdrNetBSD7}
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrNetBSD7}
	ifam := &wireFormat{extOff: sizeofIfaMsghdrNetBSD7, bodyOff: sizeofIfaMsghdrNetBSD7}
	ifanm := &wireFormat{extOff: sizeofIfAnnouncemsghdrNetBSD7, bodyOff: sizeofIfAnnouncemsghdrNetBSD7}
	// NetBSD 6 and above kernels require 64-bit aligned access to
	// routing facilities.
	return 8, map[int]parseFn{
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
		sysRTM_IFANNOUNCE: ifanm.parseInterfaceAnnounceMessage,
		sysRTM_IFINFO:     ifm.parseInterfaceMessage,
	}
}
