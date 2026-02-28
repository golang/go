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

// sizeofIfMsghdr is the size used on FreeBSD 11 for all platforms.
const sizeofIfMsghdr = 0xa8

func probeRoutingStack() (int, map[int]*wireFormat) {
	var p uintptr
	wordSize := int(unsafe.Sizeof(p))
	align := wordSize
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
	ifm := &wireFormat{extOff: 16, bodyOff: sizeofIfMsghdr}
	ifm.parse = ifm.parseInterfaceMessage
	ifam := &wireFormat{extOff: syscall.SizeofIfaMsghdr, bodyOff: syscall.SizeofIfaMsghdr}
	ifam.parse = ifam.parseInterfaceAddrMessage
	ifmam := &wireFormat{extOff: syscall.SizeofIfmaMsghdr, bodyOff: syscall.SizeofIfmaMsghdr}
	ifmam.parse = ifmam.parseInterfaceMulticastAddrMessage
	return align, map[int]*wireFormat{
		syscall.RTM_NEWADDR:  ifam,
		syscall.RTM_DELADDR:  ifam,
		syscall.RTM_IFINFO:   ifm,
		syscall.RTM_NEWMADDR: ifmam,
		syscall.RTM_DELMADDR: ifmam,
	}
}
