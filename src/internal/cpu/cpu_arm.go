// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 32

// arm doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are initialized by archauxv() and should not be changed after they are
// initialized.
var HWCap uint
var HWCap2 uint
var Platform string

// HWCAP/HWCAP2 bits. These are exposed by Linux and FreeBSD.
const (
	hwcap_VFPv4 = 1 << 16
	hwcap_IDIVA = 1 << 17
	hwcap_LPAE  = 1 << 20
)

func doinit() {
	options = []option{
		{Name: "vfpv4", Feature: &ARM.HasVFPv4},
		{Name: "idiva", Feature: &ARM.HasIDIVA},
		{Name: "v7atomics", Feature: &ARM.HasV7Atomics},
	}

	// HWCAP feature bits
	ARM.HasVFPv4 = isSet(HWCap, hwcap_VFPv4)
	ARM.HasIDIVA = isSet(HWCap, hwcap_IDIVA)
	// lpae is required to make the 64-bit instructions LDRD and STRD (and variants) atomic.
	// See ARMv7 manual section B1.6.
	// We also need at least a v7 chip, for the DMB instruction.
	ARM.HasV7Atomics = isSet(HWCap, hwcap_LPAE) && isV7(Platform)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}

func isV7(s string) bool {
	if s == "aarch64" {
		return true
	}
	return s >= "v7" // will be something like v5, v7, v8, v8l
}
