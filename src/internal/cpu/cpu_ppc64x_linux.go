// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package cpu

// ppc64 doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are initialized by archauxv and should not be changed after they are
// initialized.
var HWCap uint
var HWCap2 uint

// HWCAP bits. These are exposed by Linux.
const (
	// ISA Level
	hwcap2_ARCH_3_00 = 0x00800000

	// CPU features
	hwcap2_DARN = 0x00200000
	hwcap2_SCV  = 0x00100000
)

func osinit() {
	PPC64.IsPOWER9 = isSet(HWCap2, hwcap2_ARCH_3_00)
	PPC64.HasDARN = isSet(HWCap2, hwcap2_DARN)
	PPC64.HasSCV = isSet(HWCap2, hwcap2_SCV)
}
