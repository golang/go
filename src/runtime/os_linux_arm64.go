// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64

package runtime

// For go:linkname
import _ "unsafe"

var randomNumber uint32

// arm64 doesn't have a 'cpuid' instruction equivalent and relies on
// HWCAP/HWCAP2 bits for hardware capabilities.

//go:linkname cpu_hwcap internal/cpu.arm64_hwcap
//go:linkname cpu_hwcap2 internal/cpu.arm64_hwcap2
var cpu_hwcap uint
var cpu_hwcap2 uint

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_RANDOM:
		// sysargs filled in startupRandomData, but that
		// pointer may not be word aligned, so we must treat
		// it as a byte array.
		randomNumber = uint32(startupRandomData[4]) | uint32(startupRandomData[5])<<8 |
			uint32(startupRandomData[6])<<16 | uint32(startupRandomData[7])<<24
	case _AT_HWCAP:
		cpu_hwcap = uint(val)
	case _AT_HWCAP2:
		cpu_hwcap2 = uint(val)
	}
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// randomNumber provides better seeding of fastrand.
	return nanotime() + int64(randomNumber)
}
