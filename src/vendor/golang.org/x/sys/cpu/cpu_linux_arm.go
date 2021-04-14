// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

func doinit() {
	ARM.HasSWP = isSet(hwCap, hwcap_SWP)
	ARM.HasHALF = isSet(hwCap, hwcap_HALF)
	ARM.HasTHUMB = isSet(hwCap, hwcap_THUMB)
	ARM.Has26BIT = isSet(hwCap, hwcap_26BIT)
	ARM.HasFASTMUL = isSet(hwCap, hwcap_FAST_MULT)
	ARM.HasFPA = isSet(hwCap, hwcap_FPA)
	ARM.HasVFP = isSet(hwCap, hwcap_VFP)
	ARM.HasEDSP = isSet(hwCap, hwcap_EDSP)
	ARM.HasJAVA = isSet(hwCap, hwcap_JAVA)
	ARM.HasIWMMXT = isSet(hwCap, hwcap_IWMMXT)
	ARM.HasCRUNCH = isSet(hwCap, hwcap_CRUNCH)
	ARM.HasTHUMBEE = isSet(hwCap, hwcap_THUMBEE)
	ARM.HasNEON = isSet(hwCap, hwcap_NEON)
	ARM.HasVFPv3 = isSet(hwCap, hwcap_VFPv3)
	ARM.HasVFPv3D16 = isSet(hwCap, hwcap_VFPv3D16)
	ARM.HasTLS = isSet(hwCap, hwcap_TLS)
	ARM.HasVFPv4 = isSet(hwCap, hwcap_VFPv4)
	ARM.HasIDIVA = isSet(hwCap, hwcap_IDIVA)
	ARM.HasIDIVT = isSet(hwCap, hwcap_IDIVT)
	ARM.HasVFPD32 = isSet(hwCap, hwcap_VFPD32)
	ARM.HasLPAE = isSet(hwCap, hwcap_LPAE)
	ARM.HasEVTSTRM = isSet(hwCap, hwcap_EVTSTRM)
	ARM.HasAES = isSet(hwCap2, hwcap2_AES)
	ARM.HasPMULL = isSet(hwCap2, hwcap2_PMULL)
	ARM.HasSHA1 = isSet(hwCap2, hwcap2_SHA1)
	ARM.HasSHA2 = isSet(hwCap2, hwcap2_SHA2)
	ARM.HasCRC32 = isSet(hwCap2, hwcap2_CRC32)
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
