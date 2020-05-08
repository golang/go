// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

func (typ RIBType) parseable() bool { return true }

// RouteMetrics represents route metrics.
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

// RouteMetrics represents route metrics.
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

func probeRoutingStack() (int, map[int]*wireFormat) {
	rtm := &wireFormat{extOff: 40, bodyOff: sizeofRtMsghdrNetBSD7}
	rtm.parse = rtm.parseRouteMessage
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrNetBSD7}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: sizeofIfaMsghdrNetBSD7, bodyOff: sizeofIfaMsghdrNetBSD7}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifanm := &wireFormat{extOff: sizeofIfAnnouncemsghdrNetBSD7, bodyOff: sizeofIfAnnouncemsghdrNetBSD7}
	ifanm.parse = ifanm.parseInterfaceAnnounceMessage
	// NetBSD 6 and above kernels require 64-bit aligned access to
	// routing facilities.
	return 8, map[int]*wireFormat{
		sysRTM_ADD:        rtm,
		sysRTM_DELETE:     rtm,
		sysRTM_CHANGE:     rtm,
		sysRTM_GET:        rtm,
		sysRTM_LOSING:     rtm,
		sysRTM_REDIRECT:   rtm,
		sysRTM_MISS:       rtm,
		sysRTM_LOCK:       rtm,
		sysRTM_RESOLVE:    rtm,
		sysRTM_NEWADDR:    ifam,
		sysRTM_DELADDR:    ifam,
		sysRTM_IFANNOUNCE: ifanm,
		sysRTM_IFINFO:     ifm,
	}
}
