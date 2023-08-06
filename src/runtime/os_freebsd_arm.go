// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"unsafe"
)

const (
	_HWCAP_VFP   = 1 << 6
	_HWCAP_VFPv3 = 1 << 13
)

func checkgoarm() {
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

	// osinit not called yet, so ncpu not set: must use getncpu directly.
	if getncpu() > 1 && goarm < 7 {
		print("runtime: this system has multiple CPUs and must use\n")
		print("atomic synchronization instructions. Recompile using GOARM=7.\n")
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

//go:nosplit
func cputicks() int64 {
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}
