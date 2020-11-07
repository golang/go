// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 64

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
)

func doinit() {
	options = []option{
		{Name: "aes", Feature: &ARM64.HasAES},
		{Name: "pmull", Feature: &ARM64.HasPMULL},
		{Name: "sha1", Feature: &ARM64.HasSHA1},
		{Name: "sha2", Feature: &ARM64.HasSHA2},
		{Name: "crc32", Feature: &ARM64.HasCRC32},
		{Name: "atomics", Feature: &ARM64.HasATOMICS},
		{Name: "cpuid", Feature: &ARM64.HasCPUID},
		{Name: "isNeoverseN1", Feature: &ARM64.IsNeoverseN1},
		{Name: "isZeus", Feature: &ARM64.IsZeus},
	}

	switch GOOS {
	case "linux", "android":
		// HWCap was populated by the runtime from the auxillary vector.
		// Use HWCap information since reading aarch64 system registers
		// is not supported in user space on older linux kernels.
		ARM64.HasAES = isSet(HWCap, hwcap_AES)
		ARM64.HasPMULL = isSet(HWCap, hwcap_PMULL)
		ARM64.HasSHA1 = isSet(HWCap, hwcap_SHA1)
		ARM64.HasSHA2 = isSet(HWCap, hwcap_SHA2)
		ARM64.HasCRC32 = isSet(HWCap, hwcap_CRC32)
		ARM64.HasCPUID = isSet(HWCap, hwcap_CPUID)

		// The Samsung S9+ kernel reports support for atomics, but not all cores
		// actually support them, resulting in SIGILL. See issue #28431.
		// TODO(elias.naur): Only disable the optimization on bad chipsets on android.
		ARM64.HasATOMICS = isSet(HWCap, hwcap_ATOMICS) && GOOS != "android"

		// Check to see if executing on a NeoverseN1 and in order to do that,
		// check the AUXV for the CPUID bit. The getMIDR function executes an
		// instruction which would normally be an illegal instruction, but it's
		// trapped by the kernel, the value sanitized and then returned. Without
		// the CPUID bit the kernel will not trap the instruction and the process
		// will be terminated with SIGILL.
		if ARM64.HasCPUID {
			midr := getMIDR()
			part_num := uint16((midr >> 4) & 0xfff)
			implementor := byte((midr >> 24) & 0xff)

			if implementor == 'A' && part_num == 0xd0c {
				ARM64.IsNeoverseN1 = true
			}
			if implementor == 'A' && part_num == 0xd40 {
				ARM64.IsZeus = true
			}
		}

	case "freebsd":
		// Retrieve info from system register ID_AA64ISAR0_EL1.
		isar0 := getisar0()

		// ID_AA64ISAR0_EL1
		switch extractBits(isar0, 4, 7) {
		case 1:
			ARM64.HasAES = true
		case 2:
			ARM64.HasAES = true
			ARM64.HasPMULL = true
		}

		switch extractBits(isar0, 8, 11) {
		case 1:
			ARM64.HasSHA1 = true
		}

		switch extractBits(isar0, 12, 15) {
		case 1, 2:
			ARM64.HasSHA2 = true
		}

		switch extractBits(isar0, 16, 19) {
		case 1:
			ARM64.HasCRC32 = true
		}

		switch extractBits(isar0, 20, 23) {
		case 2:
			ARM64.HasATOMICS = true
		}
	default:
		// Other operating systems do not support reading HWCap from auxillary vector
		// or reading privileged aarch64 system registers in user space.
	}
}

func extractBits(data uint64, start, end uint) uint {
	return (uint)(data>>start) & ((1 << (end - start + 1)) - 1)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}

func getisar0() uint64

func getMIDR() uint64
