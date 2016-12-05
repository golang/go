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

// IsS390xRLD reports whether the op (as defined by an s390x.A* constant) is
// one of the RLD-like instructions that require special handling.
// The FMADD-like instructions behave similarly.
func IsS390xRLD(op obj.As) bool {
	switch op {
	case s390x.AFMADD,
		s390x.AFMADDS,
		s390x.AFMSUB,
		s390x.AFMSUBS,
		s390x.AFNMADD,
		s390x.AFNMADDS,
		s390x.AFNMSUB,
		s390x.AFNMSUBS:
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

// IsS390xWithLength reports whether the op (as defined by an s390x.A* constant)
// refers to an instruction which takes a length as its first argument.
func IsS390xWithLength(op obj.As) bool {
	switch op {
	case s390x.AMVC, s390x.ACLC, s390x.AXC, s390x.AOC, s390x.ANC:
		return true
	case s390x.AVLL, s390x.AVSTL:
		return true
	}
	return false
}

// IsS390xWithIndex reports whether the op (as defined by an s390x.A* constant)
// refers to an instruction which takes an index as its first argument.
func IsS390xWithIndex(op obj.As) bool {
	switch op {
	case s390x.AVSCEG, s390x.AVSCEF, s390x.AVGEG, s390x.AVGEF:
		return true
	case s390x.AVGMG, s390x.AVGMF, s390x.AVGMH, s390x.AVGMB:
		return true
	case s390x.AVLEIG, s390x.AVLEIF, s390x.AVLEIH, s390x.AVLEIB:
		return true
	case s390x.AVLEG, s390x.AVLEF, s390x.AVLEH, s390x.AVLEB:
		return true
	case s390x.AVSTEG, s390x.AVSTEF, s390x.AVSTEH, s390x.AVSTEB:
		return true
	case s390x.AVPDI:
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
