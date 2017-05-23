// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

func (typ RIBType) parseable() bool {
	switch typ {
	case sysNET_RT_STAT, sysNET_RT_TRASH:
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
			PathMTU: int(nativeEndian.Uint32(m.raw[m.extOff+4 : m.extOff+8])),
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
	rtm := &wireFormat{extOff: 36, bodyOff: sizeofRtMsghdrDarwin15}
	rtm2 := &wireFormat{extOff: 36, bodyOff: sizeofRtMsghdr2Darwin15}
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrDarwin15}
	ifm2 := &wireFormat{extOff: 32, bodyOff: sizeofIfMsghdr2Darwin15}
	ifam := &wireFormat{extOff: sizeofIfaMsghdrDarwin15, bodyOff: sizeofIfaMsghdrDarwin15}
	ifmam := &wireFormat{extOff: sizeofIfmaMsghdrDarwin15, bodyOff: sizeofIfmaMsghdrDarwin15}
	ifmam2 := &wireFormat{extOff: sizeofIfmaMsghdr2Darwin15, bodyOff: sizeofIfmaMsghdr2Darwin15}
	// Darwin kernels require 32-bit aligned access to routing facilities.
	return 4, map[int]parseFn{
		sysRTM_ADD:       rtm.parseRouteMessage,
		sysRTM_DELETE:    rtm.parseRouteMessage,
		sysRTM_CHANGE:    rtm.parseRouteMessage,
		sysRTM_GET:       rtm.parseRouteMessage,
		sysRTM_LOSING:    rtm.parseRouteMessage,
		sysRTM_REDIRECT:  rtm.parseRouteMessage,
		sysRTM_MISS:      rtm.parseRouteMessage,
		sysRTM_LOCK:      rtm.parseRouteMessage,
		sysRTM_RESOLVE:   rtm.parseRouteMessage,
		sysRTM_NEWADDR:   ifam.parseInterfaceAddrMessage,
		sysRTM_DELADDR:   ifam.parseInterfaceAddrMessage,
		sysRTM_IFINFO:    ifm.parseInterfaceMessage,
		sysRTM_NEWMADDR:  ifmam.parseInterfaceMulticastAddrMessage,
		sysRTM_DELMADDR:  ifmam.parseInterfaceMulticastAddrMessage,
		sysRTM_IFINFO2:   ifm2.parseInterfaceMessage,
		sysRTM_NEWMADDR2: ifmam2.parseInterfaceMulticastAddrMessage,
		sysRTM_GET2:      rtm2.parseRouteMessage,
	}
}
