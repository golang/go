// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 64

// arm64 doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are initialized by archauxv and should not be changed after they are
// initialized.
var HWCap uint
var HWCap2 uint

// HWCAP/HWCAP2 bits. These are exposed by Linux.
const (
	hwcap_AES     = 1 << 3
	hwcap_PMULL   = 1 << 4
	hwcap_SHA1    = 1 << 5
	hwcap_SHA2    = 1 << 6
	hwcap_CRC32   = 1 << 7
	hwcap_ATOMICS = 1 << 8
)

func doinit() {
	options = []option{
		{Name: "aes", Feature: &ARM64.HasAES},
		{Name: "pmull", Feature: &ARM64.HasPMULL},
		{Name: "sha1", Feature: &ARM64.HasSHA1},
		{Name: "sha2", Feature: &ARM64.HasSHA2},
		{Name: "crc32", Feature: &ARM64.HasCRC32},
		{Name: "atomics", Feature: &ARM64.HasATOMICS},
	}

	// HWCAP feature bits
	ARM64.HasAES = isSet(HWCap, hwcap_AES)
	ARM64.HasPMULL = isSet(HWCap, hwcap_PMULL)
	ARM64.HasSHA1 = isSet(HWCap, hwcap_SHA1)
	ARM64.HasSHA2 = isSet(HWCap, hwcap_SHA2)
	ARM64.HasCRC32 = isSet(HWCap, hwcap_CRC32)
	ARM64.HasATOMICS = isSet(HWCap, hwcap_ATOMICS)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
