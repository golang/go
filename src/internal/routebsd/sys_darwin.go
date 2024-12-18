// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import "syscall"

// MTU returns the interface MTU.
func (m *InterfaceMessage) MTU() int {
	return int(nativeEndian.Uint32(m.raw[m.extOff+8 : m.extOff+12]))
}

// sizeofIfMsghdr2 is copied from x/sys/unix.
const sizeofIfMsghdr2 = 0xa0

func probeRoutingStack() (int, map[int]*wireFormat) {
	ifm := &wireFormat{extOff: 16, bodyOff: syscall.SizeofIfMsghdr}
	ifm.parse = ifm.parseInterfaceMessage
	ifm2 := &wireFormat{extOff: 32, bodyOff: sizeofIfMsghdr2}
	ifm2.parse = ifm2.parseInterfaceMessage
	ifam := &wireFormat{extOff: syscall.SizeofIfaMsghdr, bodyOff: syscall.SizeofIfaMsghdr}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifmam := &wireFormat{extOff: syscall.SizeofIfmaMsghdr, bodyOff: syscall.SizeofIfmaMsghdr}
	ifmam.parse = ifmam.parseInterfaceMulticastAddrMessage
	ifmam2 := &wireFormat{extOff: syscall.SizeofIfmaMsghdr2, bodyOff: syscall.SizeofIfmaMsghdr2}
	ifmam2.parse = ifmam2.parseInterfaceMulticastAddrMessage
	// Darwin kernels require 32-bit aligned access to routing facilities.
	return 4, map[int]*wireFormat{
		syscall.RTM_NEWADDR:   ifam,
		syscall.RTM_DELADDR:   ifam,
		syscall.RTM_IFINFO:    ifm,
		syscall.RTM_NEWMADDR:  ifmam,
		syscall.RTM_DELMADDR:  ifmam,
		syscall.RTM_IFINFO2:   ifm2,
		syscall.RTM_NEWMADDR2: ifmam2,
	}
}
