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
