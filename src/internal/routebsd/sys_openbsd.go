// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import (
	"syscall"
	"unsafe"
)

func (typ RIBType) parseable() bool {
	switch typ {
	case syscall.NET_RT_STATS, syscall.NET_RT_TABLE:
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
		syscall.RTM_ADD:        rtm,
		syscall.RTM_DELETE:     rtm,
		syscall.RTM_CHANGE:     rtm,
		syscall.RTM_GET:        rtm,
		syscall.RTM_LOSING:     rtm,
		syscall.RTM_REDIRECT:   rtm,
		syscall.RTM_MISS:       rtm,
		syscall.RTM_RESOLVE:    rtm,
		syscall.RTM_NEWADDR:    ifam,
		syscall.RTM_DELADDR:    ifam,
		syscall.RTM_IFINFO:     ifm,
		syscall.RTM_IFANNOUNCE: ifanm,
		syscall.RTM_DESYNC:     rtm,
	}
}
