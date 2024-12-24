// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the
// 64-bit PowerPC (PPC64) instruction set, to minimize its interaction
// with the core of the assembler.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
)

func jumpPPC64(word string) bool {
	switch word {
	case "BC", "BCL", "BEQ", "BGE", "BGT", "BL", "BLE", "BLT", "BNE", "BR", "BVC", "BVS", "BDNZ", "BDZ", "CALL", "JMP":
		return true
	}
	return false
}

// IsPPC64CMP reports whether the op (as defined by an ppc64.A* constant) is
// one of the CMP instructions that require special handling.
func IsPPC64CMP(op obj.As) bool {
	switch op {
	case ppc64.ACMP, ppc64.ACMPU, ppc64.ACMPW, ppc64.ACMPWU, ppc64.AFCMPO, ppc64.AFCMPU, ppc64.ADCMPO, ppc64.ADCMPU, ppc64.ADCMPOQ, ppc64.ADCMPUQ:
		return true
	}
	return false
}

// IsPPC64NEG reports whether the op (as defined by an ppc64.A* constant) is
// one of the NEG-like instructions that require special handling.
func IsPPC64NEG(op obj.As) bool {
	switch op {
	case ppc64.AADDMECC, ppc64.AADDMEVCC, ppc64.AADDMEV, ppc64.AADDME,
		ppc64.AADDZECC, ppc64.AADDZEVCC, ppc64.AADDZEV, ppc64.AADDZE,
		ppc64.ACNTLZDCC, ppc64.ACNTLZD, ppc64.ACNTLZWCC, ppc64.ACNTLZW,
		ppc64.AEXTSBCC, ppc64.AEXTSB, ppc64.AEXTSHCC, ppc64.AEXTSH,
		ppc64.AEXTSWCC, ppc64.AEXTSW, ppc64.ANEGCC, ppc64.ANEGVCC,
		ppc64.ANEGV, ppc64.ANEG, ppc64.ASLBMFEE, ppc64.ASLBMFEV,
		ppc64.ASLBMTE, ppc64.ASUBMECC, ppc64.ASUBMEVCC, ppc64.ASUBMEV,
		ppc64.ASUBME, ppc64.ASUBZECC, ppc64.ASUBZEVCC, ppc64.ASUBZEV,
		ppc64.ASUBZE:
		return true
	}
	return false
}

func ppc64RegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "CR":
		if 0 <= n && n <= 7 {
			return ppc64.REG_CR0 + n, true
		}
	case "A":
		if 0 <= n && n <= 8 {
			return ppc64.REG_A0 + n, true
		}
	case "VS":
		if 0 <= n && n <= 63 {
			return ppc64.REG_VS0 + n, true
		}
	case "V":
		if 0 <= n && n <= 31 {
			return ppc64.REG_V0 + n, true
		}
	case "F":
		if 0 <= n && n <= 31 {
			return ppc64.REG_F0 + n, true
		}
	case "R":
		if 0 <= n && n <= 31 {
			return ppc64.REG_R0 + n, true
		}
	case "SPR":
		if 0 <= n && n <= 1024 {
			return ppc64.REG_SPR0 + n, true
		}
	}
	return 0, false
}
