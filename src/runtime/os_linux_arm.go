// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_AT_PLATFORM = 15 //  introduced in at least 2.6.11

	_HWCAP_VFP   = 1 << 6  // introduced in at least 2.6.11
	_HWCAP_VFPv3 = 1 << 13 // introduced in 2.6.30
	_HWCAP_IDIVA = 1 << 17
)

var randomNumber uint32
var armArch uint8 = 6 // we default to ARMv6
var hwcap uint32      // set by setup_auxv
var hardDiv bool      // set if a hardware divider is available

func checkgoarm() {
	// On Android, /proc/self/auxv might be unreadable and hwcap won't
	// reflect the CPU capabilities. Assume that every Android arm device
	// has the necessary floating point hardware available.
	if GOOS == "android" {
		return
	}
	if goarm > 5 && hwcap&_HWCAP_VFP == 0 {
		print("runtime: this CPU has no floating point hardware, so it cannot run\n")
		print("this GOARM=", goarm, " binary. Recompile using GOARM=5.\n")
		exit(1)
	}
	if goarm > 6 && hwcap&_HWCAP_VFPv3 == 0 {
		print("runtime: this CPU has no VFPv3 floating point hardware, so it cannot run\n")
		print("this GOARM=", goarm, " binary. Recompile using GOARM=5.\n")
		exit(1)
	}
}

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_RANDOM:
		// sysargs filled in startupRandomData, but that
		// pointer may not be word aligned, so we must treat
		// it as a byte array.
		randomNumber = uint32(startupRandomData[4]) | uint32(startupRandomData[5])<<8 |
			uint32(startupRandomData[6])<<16 | uint32(startupRandomData[7])<<24

	case _AT_PLATFORM: // v5l, v6l, v7l
		t := *(*uint8)(unsafe.Pointer(val + 1))
		if '5' <= t && t <= '7' {
			armArch = t - '0'
		}

	case _AT_HWCAP: // CPU capability bit flags
		hwcap = uint32(val)
		hardDiv = (hwcap & _HWCAP_IDIVA) != 0
	}
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// randomNumber provides better seeding of fastrand.
	return nanotime() + int64(randomNumber)
}
