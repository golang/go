// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const (
	// bit mask values from /usr/include/bits/hwcap.h
	hwcap_ZARCH  = 2
	hwcap_STFLE  = 4
	hwcap_MSA    = 8
	hwcap_LDISP  = 16
	hwcap_EIMM   = 32
	hwcap_DFP    = 64
	hwcap_ETF3EH = 256
	hwcap_VX     = 2048
	hwcap_VXE    = 8192
)

func initS390Xbase() {
	// test HWCAP bit vector
	has := func(featureMask uint) bool {
		return hwCap&featureMask == featureMask
	}

	// mandatory
	S390X.HasZARCH = has(hwcap_ZARCH)

	// optional
	S390X.HasSTFLE = has(hwcap_STFLE)
	S390X.HasLDISP = has(hwcap_LDISP)
	S390X.HasEIMM = has(hwcap_EIMM)
	S390X.HasETF3EH = has(hwcap_ETF3EH)
	S390X.HasDFP = has(hwcap_DFP)
	S390X.HasMSA = has(hwcap_MSA)
	S390X.HasVX = has(hwcap_VX)
	if S390X.HasVX {
		S390X.HasVXE = has(hwcap_VXE)
	}
}
