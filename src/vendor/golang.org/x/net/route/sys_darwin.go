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
			PathMTU: int(nativeEndian.Uint32(m.raw[m.extOff+4 : m.extOff+8])),
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
			Type: int(m.raw[m.extOff]),
			MTU:  int(nativeEndian.Uint32(m.raw[m.extOff+8 : m.extOff+12])),
		},
	}
}

func probeRoutingStack() (int, map[int]*wireFormat) {
	rtm := &wireFormat{extOff: 36, bodyOff: sizeofRtMsghdrDarwin15}
	rtm.parse = rtm.parseRouteMessage
	rtm2 := &wireFormat{extOff: 36, bodyOff: sizeofRtMsghdr2Darwin15}
	rtm2.parse = rtm2.parseRouteMessage
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrDarwin15}
	ifm.parse = ifm.parseInterfaceMessage
	ifm2 := &wireFormat{extOff: 32, bodyOff: sizeofIfMsghdr2Darwin15}
	ifm2.parse = ifm2.parseInterfaceMessage
	ifam := &wireFormat{extOff: sizeofIfaMsghdrDarwin15, bodyOff: sizeofIfaMsghdrDarwin15}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifmam := &wireFormat{extOff: sizeofIfmaMsghdrDarwin15, bodyOff: sizeofIfmaMsghdrDarwin15}
	ifmam.parse = ifmam.parseInterfaceMulticastAddrMessage
	ifmam2 := &wireFormat{extOff: sizeofIfmaMsghdr2Darwin15, bodyOff: sizeofIfmaMsghdr2Darwin15}
	ifmam2.parse = ifmam2.parseInterfaceMulticastAddrMessage
	// Darwin kernels require 32-bit aligned access to routing facilities.
	return 4, map[int]*wireFormat{
		sysRTM_ADD:       rtm,
		sysRTM_DELETE:    rtm,
		sysRTM_CHANGE:    rtm,
		sysRTM_GET:       rtm,
		sysRTM_LOSING:    rtm,
		sysRTM_REDIRECT:  rtm,
		sysRTM_MISS:      rtm,
		sysRTM_LOCK:      rtm,
		sysRTM_RESOLVE:   rtm,
		sysRTM_NEWADDR:   ifam,
		sysRTM_DELADDR:   ifam,
		sysRTM_IFINFO:    ifm,
		sysRTM_NEWMADDR:  ifmam,
		sysRTM_DELMADDR:  ifmam,
		sysRTM_IFINFO2:   ifm2,
		sysRTM_NEWMADDR2: ifmam2,
		sysRTM_GET2:      rtm2,
	}
}
