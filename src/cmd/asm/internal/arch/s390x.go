// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the
// s390x instruction set, to minimize its interaction
// with the core of the assembler.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
)

func jumpS390x(word string) bool {
	switch word {
	case "BC",
		"BCL",
		"BEQ",
		"BGE",
		"BGT",
		"BL",
		"BLE",
		"BLEU",
		"BLT",
		"BLTU",
		"BNE",
		"BR",
		"BVC",
		"BVS",
		"CMPBEQ",
		"CMPBGE",
		"CMPBGT",
		"CMPBLE",
		"CMPBLT",
		"CMPBNE",
		"CMPUBEQ",
		"CMPUBGE",
		"CMPUBGT",
		"CMPUBLE",
		"CMPUBLT",
		"CMPUBNE",
		"CALL",
		"JMP":
		return true
	}
	return false
}

// IsS390xCMP reports whether the op (as defined by an s390x.A* constant) is
// one of the CMP instructions that require special handling.
func IsS390xCMP(op obj.As) bool {
	switch op {
	case s390x.ACMP, s390x.ACMPU, s390x.ACMPW, s390x.ACMPWU:
		return true
	}
	return false
}

// IsS390xNEG reports whether the op (as defined by an s390x.A* constant) is
// one of the NEG-like instructions that require special handling.
func IsS390xNEG(op obj.As) bool {
	switch op {
	case s390x.ANEG, s390x.ANEGW:
		return true
	}
	return false
}

func s390xRegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "AR":
		if 0 <= n && n <= 15 {
			return s390x.REG_AR0 + n, true
		}
	case "F":
		if 0 <= n && n <= 15 {
			return s390x.REG_F0 + n, true
		}
	case "R":
		if 0 <= n && n <= 15 {
			return s390x.REG_R0 + n, true
		}
	case "V":
		if 0 <= n && n <= 31 {
			return s390x.REG_V0 + n, true
		}
	}
	return 0, false
}
