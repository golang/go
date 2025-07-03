// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

package cpu

// HWCAP/HWCAP2 bits. These are exposed by the kernel.
const (
	// ISA Level
	_PPC_FEATURE2_ARCH_2_07 = 0x80000000
	_PPC_FEATURE2_ARCH_3_00 = 0x00800000

	// CPU features
	_PPC_FEATURE2_DARN = 0x00200000
	_PPC_FEATURE2_SCV  = 0x00100000
)

func doinit() {
	// HWCAP2 feature bits
	PPC64.IsPOWER8 = isSet(hwCap2, _PPC_FEATURE2_ARCH_2_07)
	PPC64.IsPOWER9 = isSet(hwCap2, _PPC_FEATURE2_ARCH_3_00)
	PPC64.HasDARN = isSet(hwCap2, _PPC_FEATURE2_DARN)
	PPC64.HasSCV = isSet(hwCap2, _PPC_FEATURE2_SCV)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
