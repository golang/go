// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

// HWCAP/HWCAP2 bits. These are exposed by Linux.
const (
	hwcap_FP       = 1 << 0
	hwcap_ASIMD    = 1 << 1
	hwcap_EVTSTRM  = 1 << 2
	hwcap_AES      = 1 << 3
	hwcap_PMULL    = 1 << 4
	hwcap_SHA1     = 1 << 5
	hwcap_SHA2     = 1 << 6
	hwcap_CRC32    = 1 << 7
	hwcap_ATOMICS  = 1 << 8
	hwcap_FPHP     = 1 << 9
	hwcap_ASIMDHP  = 1 << 10
	hwcap_CPUID    = 1 << 11
	hwcap_ASIMDRDM = 1 << 12
	hwcap_JSCVT    = 1 << 13
	hwcap_FCMA     = 1 << 14
	hwcap_LRCPC    = 1 << 15
	hwcap_DCPOP    = 1 << 16
	hwcap_SHA3     = 1 << 17
	hwcap_SM3      = 1 << 18
	hwcap_SM4      = 1 << 19
	hwcap_ASIMDDP  = 1 << 20
	hwcap_SHA512   = 1 << 21
	hwcap_SVE      = 1 << 22
	hwcap_ASIMDFHM = 1 << 23
)

func doinit() {
	if err := readHWCAP(); err != nil {
		// failed to read /proc/self/auxv, try reading registers directly
		readARM64Registers()
		return
	}

	// HWCAP feature bits
	ARM64.HasFP = isSet(hwCap, hwcap_FP)
	ARM64.HasASIMD = isSet(hwCap, hwcap_ASIMD)
	ARM64.HasEVTSTRM = isSet(hwCap, hwcap_EVTSTRM)
	ARM64.HasAES = isSet(hwCap, hwcap_AES)
	ARM64.HasPMULL = isSet(hwCap, hwcap_PMULL)
	ARM64.HasSHA1 = isSet(hwCap, hwcap_SHA1)
	ARM64.HasSHA2 = isSet(hwCap, hwcap_SHA2)
	ARM64.HasCRC32 = isSet(hwCap, hwcap_CRC32)
	ARM64.HasATOMICS = isSet(hwCap, hwcap_ATOMICS)
	ARM64.HasFPHP = isSet(hwCap, hwcap_FPHP)
	ARM64.HasASIMDHP = isSet(hwCap, hwcap_ASIMDHP)
	ARM64.HasCPUID = isSet(hwCap, hwcap_CPUID)
	ARM64.HasASIMDRDM = isSet(hwCap, hwcap_ASIMDRDM)
	ARM64.HasJSCVT = isSet(hwCap, hwcap_JSCVT)
	ARM64.HasFCMA = isSet(hwCap, hwcap_FCMA)
	ARM64.HasLRCPC = isSet(hwCap, hwcap_LRCPC)
	ARM64.HasDCPOP = isSet(hwCap, hwcap_DCPOP)
	ARM64.HasSHA3 = isSet(hwCap, hwcap_SHA3)
	ARM64.HasSM3 = isSet(hwCap, hwcap_SM3)
	ARM64.HasSM4 = isSet(hwCap, hwcap_SM4)
	ARM64.HasASIMDDP = isSet(hwCap, hwcap_ASIMDDP)
	ARM64.HasSHA512 = isSet(hwCap, hwcap_SHA512)
	ARM64.HasSVE = isSet(hwCap, hwcap_SVE)
	ARM64.HasASIMDFHM = isSet(hwCap, hwcap_ASIMDFHM)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
