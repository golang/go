// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !amd64,!amd64p32,!386

package cpu

import (
	"io/ioutil"
)

const (
	_AT_HWCAP  = 16
	_AT_HWCAP2 = 26

	procAuxv = "/proc/self/auxv"

	uintSize = int(32 << (^uint(0) >> 63))
)

// For those platforms don't have a 'cpuid' equivalent we use HWCAP/HWCAP2
// These are initialized in cpu_$GOARCH.go
// and should not be changed after they are initialized.
var hwCap uint
var hwCap2 uint

func init() {
	buf, err := ioutil.ReadFile(procAuxv)
	if err != nil {
		// e.g. on android /proc/self/auxv is not accessible, so silently
		// ignore the error and leave Initialized = false
		return
	}

	bo := hostByteOrder()
	for len(buf) >= 2*(uintSize/8) {
		var tag, val uint
		switch uintSize {
		case 32:
			tag = uint(bo.Uint32(buf[0:]))
			val = uint(bo.Uint32(buf[4:]))
			buf = buf[8:]
		case 64:
			tag = uint(bo.Uint64(buf[0:]))
			val = uint(bo.Uint64(buf[8:]))
			buf = buf[16:]
		}
		switch tag {
		case _AT_HWCAP:
			hwCap = val
		case _AT_HWCAP2:
			hwCap2 = val
		}
	}
	doinit()

	Initialized = true
}
