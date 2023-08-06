// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"unsafe"
)

const (
	_HWCAP_VFP   = 1 << 6  // introduced in at least 2.6.11
	_HWCAP_VFPv3 = 1 << 13 // introduced in 2.6.30
)

func vdsoCall()

func checkgoarm() {
	// On Android, /proc/self/auxv might be unreadable and hwcap won't
	// reflect the CPU capabilities. Assume that every Android arm device
	// has the necessary floating point hardware available.
	if GOOS == "android" {
		return
	}
	if cpu.HWCap&_HWCAP_VFP == 0 && goarmsoftfp == 0 {
		print("runtime: this CPU has no floating point hardware, so it cannot run\n")
		print("a binary compiled for hard floating point. Recompile adding ,softfloat\n")
		print("to GOARM.\n")
		exit(1)
	}
	if goarm > 6 && cpu.HWCap&_HWCAP_VFPv3 == 0 && goarmsoftfp == 0 {
		print("runtime: this CPU has no VFPv3 floating point hardware, so it cannot run\n")
		print("a binary compiled for VFPv3 hard floating point. Recompile adding ,softfloat\n")
		print("to GOARM or changing GOARM to 6.\n")
		exit(1)
	}
}

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP:
		cpu.HWCap = uint(val)
	case _AT_HWCAP2:
		cpu.HWCap2 = uint(val)
	case _AT_PLATFORM:
		cpu.Platform = gostringnocopy((*byte)(unsafe.Pointer(val)))
	}
}

func osArchInit() {}

//go:nosplit
func cputicks() int64 {
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}
