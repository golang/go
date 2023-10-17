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

// HWCAP/HWCAP2 bits. These are exposed by Linux and FreeBSD.
const (
	hwcap_VFPv4 = 1 << 16
	hwcap_IDIVA = 1 << 17
)

func doinit() {
	options = []option{
		{Name: "vfpv4", Feature: &ARM.HasVFPv4},
		{Name: "idiva", Feature: &ARM.HasIDIVA},
	}

	// HWCAP feature bits
	ARM.HasVFPv4 = isSet(HWCap, hwcap_VFPv4)
	ARM.HasIDIVA = isSet(HWCap, hwcap_IDIVA)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
