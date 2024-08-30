// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import (
	"syscall"
	"unsafe"
)

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
	var p uintptr
	rtm := &wireFormat{extOff: 40, bodyOff: sizeofRtMsghdrDragonFlyBSD4}
	rtm.parse = rtm.parseRouteMessage
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdrDragonFlyBSD4}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: sizeofIfaMsghdrDragonFlyBSD4, bodyOff: sizeofIfaMsghdrDragonFlyBSD4}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifmam := &wireFormat{extOff: sizeofIfmaMsghdrDragonFlyBSD4, bodyOff: sizeofIfmaMsghdrDragonFlyBSD4}
	ifmam.parse = ifmam.parseInterfaceMulticastAddrMessage
	ifanm := &wireFormat{extOff: sizeofIfAnnouncemsghdrDragonFlyBSD4, bodyOff: sizeofIfAnnouncemsghdrDragonFlyBSD4}
	ifanm.parse = ifanm.parseInterfaceAnnounceMessage

	rel, _ := syscall.SysctlUint32("kern.osreldate")
	if rel >= 500705 {
		// https://github.com/DragonFlyBSD/DragonFlyBSD/commit/43a373152df2d405c9940983e584e6a25e76632d
		// but only the size of struct ifa_msghdr actually changed
		rtmVersion = 7
		ifam.bodyOff = sizeofIfaMsghdrDragonFlyBSD58
	}

	return int(unsafe.Sizeof(p)), map[int]*wireFormat{
		syscall.RTM_ADD:        rtm,
		syscall.RTM_DELETE:     rtm,
		syscall.RTM_CHANGE:     rtm,
		syscall.RTM_GET:        rtm,
		syscall.RTM_LOSING:     rtm,
		syscall.RTM_REDIRECT:   rtm,
		syscall.RTM_MISS:       rtm,
		syscall.RTM_LOCK:       rtm,
		syscall.RTM_RESOLVE:    rtm,
		syscall.RTM_NEWADDR:    ifam,
		syscall.RTM_DELADDR:    ifam,
		syscall.RTM_IFINFO:     ifm,
		syscall.RTM_NEWMADDR:   ifmam,
		syscall.RTM_DELMADDR:   ifmam,
		syscall.RTM_IFANNOUNCE: ifanm,
	}
}
