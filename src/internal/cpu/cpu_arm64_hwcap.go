// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && linux

package cpu

// HWCap may be initialized by archauxv and
// should not be changed after it was initialized.
var HWCap uint

// HWCAP bits. These are exposed by Linux.
const (
	hwcap_AES     = 1 << 3
	hwcap_PMULL   = 1 << 4
	hwcap_SHA1    = 1 << 5
	hwcap_SHA2    = 1 << 6
	hwcap_CRC32   = 1 << 7
	hwcap_ATOMICS = 1 << 8
	hwcap_CPUID   = 1 << 11
	hwcap_SHA512  = 1 << 21
)

func hwcapInit(os string) {
	// HWCap was populated by the runtime from the auxiliary vector.
	// Use HWCap information since reading aarch64 system registers
	// is not supported in user space on older linux kernels.
	ARM64.HasAES = isSet(HWCap, hwcap_AES)
	ARM64.HasPMULL = isSet(HWCap, hwcap_PMULL)
	ARM64.HasSHA1 = isSet(HWCap, hwcap_SHA1)
	ARM64.HasSHA2 = isSet(HWCap, hwcap_SHA2)
	ARM64.HasCRC32 = isSet(HWCap, hwcap_CRC32)
	ARM64.HasCPUID = isSet(HWCap, hwcap_CPUID)
	ARM64.HasSHA512 = isSet(HWCap, hwcap_SHA512)

	// The Samsung S9+ kernel reports support for atomics, but not all cores
	// actually support them, resulting in SIGILL. See issue #28431.
	// TODO(elias.naur): Only disable the optimization on bad chipsets on android.
	ARM64.HasATOMICS = isSet(HWCap, hwcap_ATOMICS) && os != "android"

	// Check to see if executing on a Neoverse core and in order to do that,
	// check the AUXV for the CPUID bit. The getMIDR function executes an
	// instruction which would normally be an illegal instruction, but it's
	// trapped by the kernel, the value sanitized and then returned.
	// Without the CPUID bit the kernel will not trap the instruction and the
	// process will be terminated with SIGILL.
	if ARM64.HasCPUID {
		midr := getMIDR()
		part_num := uint16((midr >> 4) & 0xfff)
		implementer := byte((midr >> 24) & 0xff)

		// d0c - NeoverseN1
		// d40 - NeoverseV1
		// d49 - NeoverseN2
		// d4f - NeoverseV2
		if implementer == 'A' && (part_num == 0xd0c || part_num == 0xd40 ||
			part_num == 0xd49 || part_num == 0xd4f) {
			ARM64.IsNeoverse = true
		}
	}
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
