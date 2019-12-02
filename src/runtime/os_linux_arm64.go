// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64

package runtime

import "internal/cpu"

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP:
		// arm64 doesn't have a 'cpuid' instruction equivalent and relies on
		// HWCAP/HWCAP2 bits for hardware capabilities.
		hwcap := uint(val)
		if GOOS == "android" {
			// The Samsung S9+ kernel reports support for atomics, but not all cores
			// actually support them, resulting in SIGILL. See issue #28431.
			// TODO(elias.naur): Only disable the optimization on bad chipsets.
			const hwcap_ATOMICS = 1 << 8
			hwcap &= ^uint(hwcap_ATOMICS)
		}
		cpu.HWCap = hwcap
	case _AT_HWCAP2:
		cpu.HWCap2 = uint(val)
	}
}

func osArchInit() {}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}
