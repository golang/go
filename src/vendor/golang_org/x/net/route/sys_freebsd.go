// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import (
	"syscall"
	"unsafe"
)

func (typ RIBType) parseable() bool { return true }

// A RouteMetrics represents route metrics.
type RouteMetrics struct {
	PathMTU int // path maximum transmission unit
}

// SysType implements the SysType method of Sys interface.
func (rmx *RouteMetrics) SysType() SysType { return SysMetrics }

// Sys implements the Sys method of Message interface.
func (m *RouteMessage) Sys() []Sys {
	if kernelAlign == 8 {
		return []Sys{
			&RouteMetrics{
				PathMTU: int(nativeEndian.Uint64(m.raw[m.extOff+8 : m.extOff+16])),
			},
		}
	}
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
	var p uintptr
	wordSize := int(unsafe.Sizeof(p))
	align := int(unsafe.Sizeof(p))
	// In the case of kern.supported_archs="amd64 i386", we need
	// to know the underlying kernel's architecture because the
	// alignment for routing facilities are set at the build time
	// of the kernel.
	conf, _ := syscall.Sysctl("kern.conftxt")
	for i, j := 0, 0; j < len(conf); j++ {
		if conf[j] != '\n' {
			continue
		}
		s := conf[i:j]
		i = j + 1
		if len(s) > len("machine") && s[:len("machine")] == "machine" {
			s = s[len("machine"):]
			for k := 0; k < len(s); k++ {
				if s[k] == ' ' || s[k] == '\t' {
					s = s[1:]
				}
				break
			}
			if s == "amd64" {
				align = 8
			}
			break
		}
	}
	var rtm, ifm, ifam, ifmam, ifanm *wireFormat
	if align != wordSize { // 386 emulation on amd64
		rtm = &wireFormat{extOff: sizeofRtMsghdrFreeBSD10Emu - sizeofRtMetricsFreeBSD10Emu, bodyOff: sizeofRtMsghdrFreeBSD10Emu}
		ifm = &wireFormat{extOff: 16}
		ifam = &wireFormat{extOff: sizeofIfaMsghdrFreeBSD10Emu, bodyOff: sizeofIfaMsghdrFreeBSD10Emu}
		ifmam = &wireFormat{extOff: sizeofIfmaMsghdrFreeBSD10Emu, bodyOff: sizeofIfmaMsghdrFreeBSD10Emu}
		ifanm = &wireFormat{extOff: sizeofIfAnnouncemsghdrFreeBSD10Emu, bodyOff: sizeofIfAnnouncemsghdrFreeBSD10Emu}
	} else {
		rtm = &wireFormat{extOff: sizeofRtMsghdrFreeBSD10 - sizeofRtMetricsFreeBSD10, bodyOff: sizeofRtMsghdrFreeBSD10}
		ifm = &wireFormat{extOff: 16}
		ifam = &wireFormat{extOff: sizeofIfaMsghdrFreeBSD10, bodyOff: sizeofIfaMsghdrFreeBSD10}
		ifmam = &wireFormat{extOff: sizeofIfmaMsghdrFreeBSD10, bodyOff: sizeofIfmaMsghdrFreeBSD10}
		ifanm = &wireFormat{extOff: sizeofIfAnnouncemsghdrFreeBSD10, bodyOff: sizeofIfAnnouncemsghdrFreeBSD10}
	}
	rel, _ := syscall.SysctlUint32("kern.osreldate")
	switch {
	case rel < 800000:
		if align != wordSize { // 386 emulation on amd64
			ifm.bodyOff = sizeofIfMsghdrFreeBSD7Emu
		} else {
			ifm.bodyOff = sizeofIfMsghdrFreeBSD7
		}
	case 800000 <= rel && rel < 900000:
		if align != wordSize { // 386 emulation on amd64
			ifm.bodyOff = sizeofIfMsghdrFreeBSD8Emu
		} else {
			ifm.bodyOff = sizeofIfMsghdrFreeBSD8
		}
	case 900000 <= rel && rel < 1000000:
		if align != wordSize { // 386 emulation on amd64
			ifm.bodyOff = sizeofIfMsghdrFreeBSD9Emu
		} else {
			ifm.bodyOff = sizeofIfMsghdrFreeBSD9
		}
	case 1000000 <= rel && rel < 1100000:
		if align != wordSize { // 386 emulation on amd64
			ifm.bodyOff = sizeofIfMsghdrFreeBSD10Emu
		} else {
			ifm.bodyOff = sizeofIfMsghdrFreeBSD10
		}
	default:
		if align != wordSize { // 386 emulation on amd64
			ifm.bodyOff = sizeofIfMsghdrFreeBSD11Emu
		} else {
			ifm.bodyOff = sizeofIfMsghdrFreeBSD11
		}
	}
	return align, map[int]parseFn{
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
