// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 32

// arm doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are linknamed in runtime/os_(linux|freebsd)_arm.go and are
// initialized by archauxv().
// These should not be changed after they are initialized.
var HWCap uint
var HWCap2 uint

// HWCAP/HWCAP2 bits. These are exposed by Linux and FreeBSD.
const (
	hwcap_IDIVA = 1 << 17
)

func doinit() {
	options = []option{
		{"idiva", &ARM.HasIDIVA},
	}

	// HWCAP feature bits
	ARM.HasIDIVA = isSet(HWCap, hwcap_IDIVA)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
