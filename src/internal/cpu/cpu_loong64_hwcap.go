// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64 && linux

package cpu

// This is initialized by archauxv and should not be changed after it is
// initialized.
var HWCap uint

// HWCAP bits. These are exposed by the Linux kernel.
const (
	hwcap_LOONGARCH_CRC32 = 1 << 6
)

func hwcapInit() {
	// It is not taken from CPUCFG data regardless of availability of
	// CPUCFG, because the CPUCFG data only reflects capabilities of the
	// hardware, but not kernel support.
	//
	// As of 2023, we do not know for sure if the CPUCFG data can be
	// patched in software, nor does any known LoongArch kernel do that.
	Loong64.HasCRC32 = isSet(HWCap, hwcap_LOONGARCH_CRC32)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
