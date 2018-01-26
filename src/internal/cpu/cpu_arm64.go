// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLineSize = 64

// arm64 doesn't have a 'cpuid' equivalent, so we rely on HWCAP/HWCAP2.
// These are linknamed in runtime/os_linux_arm64.go and are initialized by
// archauxv().
var hwcap uint
var hwcap2 uint

// HWCAP/HWCAP2 bits. These are exposed by Linux.
const (
	hwcap_FP       = (1 << 0)
	hwcap_ASIMD    = (1 << 1)
	hwcap_EVTSTRM  = (1 << 2)
	hwcap_AES      = (1 << 3)
	hwcap_PMULL    = (1 << 4)
	hwcap_SHA1     = (1 << 5)
	hwcap_SHA2     = (1 << 6)
	hwcap_CRC32    = (1 << 7)
	hwcap_ATOMICS  = (1 << 8)
	hwcap_FPHP     = (1 << 9)
	hwcap_ASIMDHP  = (1 << 10)
	hwcap_CPUID    = (1 << 11)
	hwcap_ASIMDRDM = (1 << 12)
	hwcap_JSCVT    = (1 << 13)
	hwcap_FCMA     = (1 << 14)
	hwcap_LRCPC    = (1 << 15)
	hwcap_DCPOP    = (1 << 16)
	hwcap_SHA3     = (1 << 17)
	hwcap_SM3      = (1 << 18)
	hwcap_SM4      = (1 << 19)
	hwcap_ASIMDDP  = (1 << 20)
	hwcap_SHA512   = (1 << 21)
	hwcap_SVE      = (1 << 22)
	hwcap_ASIMDFHM = (1 << 23)
)

func doinit() {
	options = []option{
		{"evtstrm", &ARM64.HasEVTSTRM},
		{"aes", &ARM64.HasAES},
		{"pmull", &ARM64.HasPMULL},
		{"sha1", &ARM64.HasSHA1},
		{"sha2", &ARM64.HasSHA2},
		{"crc32", &ARM64.HasCRC32},
		{"atomics", &ARM64.HasATOMICS},
		{"fphp", &ARM64.HasFPHP},
		{"asimdhp", &ARM64.HasASIMDHP},
		{"cpuid", &ARM64.HasCPUID},
		{"asimdrdm", &ARM64.HasASIMDRDM},
		{"jscvt", &ARM64.HasJSCVT},
		{"fcma", &ARM64.HasFCMA},
		{"lrcpc", &ARM64.HasLRCPC},
		{"dcpop", &ARM64.HasDCPOP},
		{"sha3", &ARM64.HasSHA3},
		{"sm3", &ARM64.HasSM3},
		{"sm4", &ARM64.HasSM4},
		{"asimddp", &ARM64.HasASIMDDP},
		{"sha512", &ARM64.HasSHA512},
		{"sve", &ARM64.HasSVE},
		{"asimdfhm", &ARM64.HasASIMDFHM},

		// These capabilities should always be enabled on arm64:
		//  {"fp", &ARM64.HasFP},
		//  {"asimd", &ARM64.HasASIMD},
	}

	// HWCAP feature bits
	ARM64.HasFP = isSet(hwcap, hwcap_FP)
	ARM64.HasASIMD = isSet(hwcap, hwcap_ASIMD)
	ARM64.HasEVTSTRM = isSet(hwcap, hwcap_EVTSTRM)
	ARM64.HasAES = isSet(hwcap, hwcap_AES)
	ARM64.HasPMULL = isSet(hwcap, hwcap_PMULL)
	ARM64.HasSHA1 = isSet(hwcap, hwcap_SHA1)
	ARM64.HasSHA2 = isSet(hwcap, hwcap_SHA2)
	ARM64.HasCRC32 = isSet(hwcap, hwcap_CRC32)
	ARM64.HasATOMICS = isSet(hwcap, hwcap_ATOMICS)
	ARM64.HasFPHP = isSet(hwcap, hwcap_FPHP)
	ARM64.HasASIMDHP = isSet(hwcap, hwcap_ASIMDHP)
	ARM64.HasCPUID = isSet(hwcap, hwcap_CPUID)
	ARM64.HasASIMDRDM = isSet(hwcap, hwcap_ASIMDRDM)
	ARM64.HasJSCVT = isSet(hwcap, hwcap_JSCVT)
	ARM64.HasFCMA = isSet(hwcap, hwcap_FCMA)
	ARM64.HasLRCPC = isSet(hwcap, hwcap_LRCPC)
	ARM64.HasDCPOP = isSet(hwcap, hwcap_DCPOP)
	ARM64.HasSHA3 = isSet(hwcap, hwcap_SHA3)
	ARM64.HasSM3 = isSet(hwcap, hwcap_SM3)
	ARM64.HasSM4 = isSet(hwcap, hwcap_SM4)
	ARM64.HasASIMDDP = isSet(hwcap, hwcap_ASIMDDP)
	ARM64.HasSHA512 = isSet(hwcap, hwcap_SHA512)
	ARM64.HasSVE = isSet(hwcap, hwcap_SVE)
	ARM64.HasASIMDFHM = isSet(hwcap, hwcap_ASIMDFHM)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
