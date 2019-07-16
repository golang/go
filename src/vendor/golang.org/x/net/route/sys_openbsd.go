// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import "unsafe"

func (typ RIBType) parseable() bool {
	switch typ {
	case sysNET_RT_STATS, sysNET_RT_TABLE:
		return false
	default:
		return true
	}
}

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
			PathMTU: int(nativeEndian.Uint32(m.raw[60:64])),
		},
	}
}

// InterfaceMetrics represents interface metrics.
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
			Type: int(m.raw[24]),
			MTU:  int(nativeEndian.Uint32(m.raw[28:32])),
		},
	}
}

func probeRoutingStack() (int, map[int]*wireFormat) {
	var p uintptr
	rtm := &wireFormat{extOff: -1, bodyOff: -1}
	rtm.parse = rtm.parseRouteMessage
	ifm := &wireFormat{extOff: -1, bodyOff: -1}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: -1, bodyOff: -1}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifanm := &wireFormat{extOff: -1, bodyOff: -1}
	ifanm.parse = ifanm.parseInterfaceAnnounceMessage
	return int(unsafe.Sizeof(p)), map[int]*wireFormat{
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
		sysRTM_IFINFO:     ifm,
		sysRTM_IFANNOUNCE: ifanm,
		sysRTM_DESYNC:     rtm,
	}
}
