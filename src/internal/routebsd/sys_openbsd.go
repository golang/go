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
	return int(nativeEndian.Uint32(m.raw[28:32]))
}

func probeRoutingStack() (int, map[int]*wireFormat) {
	var p uintptr
	ifm := &wireFormat{extOff: -1, bodyOff: -1}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: -1, bodyOff: -1}
	ifam.parse = ifam.parseInterfaceAddrMessage
	return int(unsafe.Sizeof(p)), map[int]*wireFormat{
		syscall.RTM_NEWADDR: ifam,
		syscall.RTM_DELADDR: ifam,
		syscall.RTM_IFINFO:  ifm,
	}
}
