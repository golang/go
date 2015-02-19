// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the ARM
// instruction set, to minimize its interaction with the core of the
// assembler.

package arch

import "cmd/internal/obj/ppc64"

func jumpPPC64(word string) bool {
	switch word {
	case "BC", "BCL", "BEQ", "BGE", "BGT", "BL", "BLE", "BLT", "BNE", "BR", "BVC", "BVS", "CALL":
		return true
	}
	return false
}

// IsPPC64RLD reports whether the op (as defined by an ppc64.A* constant) is
// one of the RLD-like instructions that require special handling.
func IsPPC64RLD(op int) bool {
	switch op {
	case ppc64.ARLDC, ppc64.ARLDCCC, ppc64.ARLDCL, ppc64.ARLDCLCC,
		ppc64.ARLDCR, ppc64.ARLDCRCC, ppc64.ARLDMI, ppc64.ARLDMICC,
		ppc64.ARLWMI, ppc64.ARLWMICC, ppc64.ARLWNM, ppc64.ARLWNMCC:
		return true
	}
	return false
}

// IsPPC64CMP reports whether the op (as defined by an ppc64.A* constant) is
// one of the CMP instructions that require special handling.
func IsPPC64CMP(op int) bool {
	switch op {
	case ppc64.ACMP, ppc64.ACMPU, ppc64.ACMPW, ppc64.ACMPWU:
		return true
	}
	return false
}

func ppc64RegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "CR":
		if 0 <= n && n <= 7 {
			return ppc64.REG_C0 + n, true
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
