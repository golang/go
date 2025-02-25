// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import (
	"syscall"
	"unsafe"
)

// MTU returns the interface MTU.
func (m *InterfaceMessage) MTU() int {
	return int(nativeEndian.Uint32(m.raw[m.extOff+8 : m.extOff+12]))
}

func probeRoutingStack() (int, map[int]*wireFormat) {
	var p uintptr
	ifm := &wireFormat{extOff: 16, bodyOff: syscall.SizeofIfMsghdr}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: syscall.SizeofIfaMsghdr, bodyOff: syscall.SizeofIfaMsghdr}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifmam := &wireFormat{extOff: syscall.SizeofIfmaMsghdr, bodyOff: syscall.SizeofIfmaMsghdr}
	ifmam.parse = ifmam.parseInterfaceMulticastAddrMessage

	rel, _ := syscall.SysctlUint32("kern.osreldate")
	if rel >= 500705 {
		// https://github.com/DragonFlyBSD/DragonFlyBSD/commit/43a373152df2d405c9940983e584e6a25e76632d
		// but only the size of struct ifa_msghdr actually changed
		rtmVersion = 7
		ifam.bodyOff = 0x18
	}

	return int(unsafe.Sizeof(p)), map[int]*wireFormat{
		syscall.RTM_NEWADDR:  ifam,
		syscall.RTM_DELADDR:  ifam,
		syscall.RTM_IFINFO:   ifm,
		syscall.RTM_NEWMADDR: ifmam,
		syscall.RTM_DELMADDR: ifmam,
	}
}
