// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const cacheLineSize = 32

// HWCAP/HWCAP2 bits.
// These are specific to Linux.
const (
	hwcap_SWP       = 1 << 0
	hwcap_HALF      = 1 << 1
	hwcap_THUMB     = 1 << 2
	hwcap_26BIT     = 1 << 3
	hwcap_FAST_MULT = 1 << 4
	hwcap_FPA       = 1 << 5
	hwcap_VFP       = 1 << 6
	hwcap_EDSP      = 1 << 7
	hwcap_JAVA      = 1 << 8
	hwcap_IWMMXT    = 1 << 9
	hwcap_CRUNCH    = 1 << 10
	hwcap_THUMBEE   = 1 << 11
	hwcap_NEON      = 1 << 12
	hwcap_VFPv3     = 1 << 13
	hwcap_VFPv3D16  = 1 << 14
	hwcap_TLS       = 1 << 15
	hwcap_VFPv4     = 1 << 16
	hwcap_IDIVA     = 1 << 17
	hwcap_IDIVT     = 1 << 18
	hwcap_VFPD32    = 1 << 19
	hwcap_LPAE      = 1 << 20
	hwcap_EVTSTRM   = 1 << 21

	hwcap2_AES   = 1 << 0
	hwcap2_PMULL = 1 << 1
	hwcap2_SHA1  = 1 << 2
	hwcap2_SHA2  = 1 << 3
	hwcap2_CRC32 = 1 << 4
)

func initOptions() {
	options = []option{
		{Name: "pmull", Feature: &ARM.HasPMULL},
		{Name: "sha1", Feature: &ARM.HasSHA1},
		{Name: "sha2", Feature: &ARM.HasSHA2},
		{Name: "swp", Feature: &ARM.HasSWP},
		{Name: "thumb", Feature: &ARM.HasTHUMB},
		{Name: "thumbee", Feature: &ARM.HasTHUMBEE},
		{Name: "tls", Feature: &ARM.HasTLS},
		{Name: "vfp", Feature: &ARM.HasVFP},
		{Name: "vfpd32", Feature: &ARM.HasVFPD32},
		{Name: "vfpv3", Feature: &ARM.HasVFPv3},
		{Name: "vfpv3d16", Feature: &ARM.HasVFPv3D16},
		{Name: "vfpv4", Feature: &ARM.HasVFPv4},
		{Name: "half", Feature: &ARM.HasHALF},
		{Name: "26bit", Feature: &ARM.Has26BIT},
		{Name: "fastmul", Feature: &ARM.HasFASTMUL},
		{Name: "fpa", Feature: &ARM.HasFPA},
		{Name: "edsp", Feature: &ARM.HasEDSP},
		{Name: "java", Feature: &ARM.HasJAVA},
		{Name: "iwmmxt", Feature: &ARM.HasIWMMXT},
		{Name: "crunch", Feature: &ARM.HasCRUNCH},
		{Name: "neon", Feature: &ARM.HasNEON},
		{Name: "idivt", Feature: &ARM.HasIDIVT},
		{Name: "idiva", Feature: &ARM.HasIDIVA},
		{Name: "lpae", Feature: &ARM.HasLPAE},
		{Name: "evtstrm", Feature: &ARM.HasEVTSTRM},
		{Name: "aes", Feature: &ARM.HasAES},
		{Name: "crc32", Feature: &ARM.HasCRC32},
	}

}
