// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64

package cpu

const CacheLineSize = 64

// arm64 doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are linknamed in runtime/os_linux_arm64.go and are initialized by
// archauxv().
var arm64_hwcap uint
var arm64_hwcap2 uint

// HWCAP/HWCAP2 bits. These are exposed by Linux.
const (
	hwcap_FP      = (1 << 0)
	hwcap_ASIMD   = (1 << 1)
	hwcap_EVTSTRM = (1 << 2)
	hwcap_AES     = (1 << 3)
	hwcap_PMULL   = (1 << 4)
	hwcap_SHA1    = (1 << 5)
	hwcap_SHA2    = (1 << 6)
	hwcap_CRC32   = (1 << 7)
	hwcap_ATOMICS = (1 << 8)
)

func init() {
	// HWCAP feature bits
	ARM64.HasFP = isSet(arm64_hwcap, hwcap_FP)
	ARM64.HasASIMD = isSet(arm64_hwcap, hwcap_ASIMD)
	ARM64.HasEVTSTRM = isSet(arm64_hwcap, hwcap_EVTSTRM)
	ARM64.HasAES = isSet(arm64_hwcap, hwcap_AES)
	ARM64.HasPMULL = isSet(arm64_hwcap, hwcap_PMULL)
	ARM64.HasSHA1 = isSet(arm64_hwcap, hwcap_SHA1)
	ARM64.HasSHA2 = isSet(arm64_hwcap, hwcap_SHA2)
	ARM64.HasCRC32 = isSet(arm64_hwcap, hwcap_CRC32)
	ARM64.HasATOMICS = isSet(arm64_hwcap, hwcap_ATOMICS)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
