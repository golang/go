// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package cpu

const CacheLinePadSize = 128

// ppc64x doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are initialized by archauxv in runtime/os_linux_ppc64x.go.
// These should not be changed after they are initialized.
// On aix/ppc64, these values are initialized early in the runtime in runtime/os_aix.go.
var HWCap uint
var HWCap2 uint

// HWCAP/HWCAP2 bits. These are exposed by the kernel.
const (
	// ISA Level
	PPC_FEATURE2_ARCH_2_07 = 0x80000000
	PPC_FEATURE2_ARCH_3_00 = 0x00800000

	// CPU features
	PPC_FEATURE2_DARN = 0x00200000
	PPC_FEATURE2_SCV  = 0x00100000
)

func doinit() {
	options = []option{
		{Name: "darn", Feature: &PPC64.HasDARN},
		{Name: "scv", Feature: &PPC64.HasSCV},
		{Name: "power9", Feature: &PPC64.IsPOWER9},

		// These capabilities should always be enabled on ppc64 and ppc64le:
		{Name: "power8", Feature: &PPC64.IsPOWER8, Required: true},
	}

	// HWCAP2 feature bits
	PPC64.IsPOWER8 = isSet(HWCap2, PPC_FEATURE2_ARCH_2_07)
	PPC64.IsPOWER9 = isSet(HWCap2, PPC_FEATURE2_ARCH_3_00)
	PPC64.HasDARN = isSet(HWCap2, PPC_FEATURE2_DARN)
	PPC64.HasSCV = isSet(HWCap2, PPC_FEATURE2_SCV)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
