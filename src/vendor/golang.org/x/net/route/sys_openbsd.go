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
			PathMTU: int(nativeEndian.Uint32(m.raw[60:64])),
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
			Type: int(m.raw[24]),
			MTU:  int(nativeEndian.Uint32(m.raw[28:32])),
		},
	}
}

func probeRoutingStack() (int, map[int]parseFn) {
	var p uintptr
	nooff := &wireFormat{extOff: -1, bodyOff: -1}
	return int(unsafe.Sizeof(p)), map[int]parseFn{
		sysRTM_ADD:        nooff.parseRouteMessage,
		sysRTM_DELETE:     nooff.parseRouteMessage,
		sysRTM_CHANGE:     nooff.parseRouteMessage,
		sysRTM_GET:        nooff.parseRouteMessage,
		sysRTM_LOSING:     nooff.parseRouteMessage,
		sysRTM_REDIRECT:   nooff.parseRouteMessage,
		sysRTM_MISS:       nooff.parseRouteMessage,
		sysRTM_LOCK:       nooff.parseRouteMessage,
		sysRTM_RESOLVE:    nooff.parseRouteMessage,
		sysRTM_NEWADDR:    nooff.parseInterfaceAddrMessage,
		sysRTM_DELADDR:    nooff.parseInterfaceAddrMessage,
		sysRTM_IFINFO:     nooff.parseInterfaceMessage,
		sysRTM_IFANNOUNCE: nooff.parseInterfaceAnnounceMessage,
	}
}
