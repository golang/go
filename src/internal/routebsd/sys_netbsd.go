// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import "syscall"

// MTU returns the interface MTU.
func (m *InterfaceMessage) MTU() int {
	return int(nativeEndian.Uint32(m.raw[m.extOff+8 : m.extOff+12]))
}

func probeRoutingStack() (int, map[int]*wireFormat) {
	ifm := &wireFormat{extOff: 16, bodyOff: syscall.SizeofIfMsghdr}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: syscall.SizeofIfaMsghdr, bodyOff: syscall.SizeofIfaMsghdr}
	ifam.parse = ifam.parseInterfaceAddrMessage
	// NetBSD 6 and above kernels require 64-bit aligned access to
	// routing facilities.
	return 8, map[int]*wireFormat{
		syscall.RTM_NEWADDR: ifam,
		syscall.RTM_DELADDR: ifam,
		syscall.RTM_IFINFO:  ifm,
	}
}
