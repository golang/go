// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the ARM64
// instruction set, to minimize its interaction with the core of the
// assembler.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"errors"
)

var arm64LS = map[string]uint8{
	"P": arm64.C_XPOST,
	"W": arm64.C_XPRE,
}

var arm64Jump = map[string]bool{
	"B":     true,
	"BL":    true,
	"BEQ":   true,
	"BNE":   true,
	"BCS":   true,
	"BHS":   true,
	"BCC":   true,
	"BLO":   true,
	"BMI":   true,
	"BPL":   true,
	"BVS":   true,
	"BVC":   true,
	"BHI":   true,
	"BLS":   true,
	"BGE":   true,
	"BLT":   true,
	"BGT":   true,
	"BLE":   true,
	"CALL":  true,
	"CBZ":   true,
	"CBZW":  true,
	"CBNZ":  true,
	"CBNZW": true,
	"JMP":   true,
	"TBNZ":  true,
	"TBZ":   true,

	// ADR isn't really a jump, but it takes a PC or label reference,
	// which needs to patched like a jump.
	"ADR":  true,
	"ADRP": true,
}

func jumpArm64(word string) bool {
	return arm64Jump[word]
}

var arm64SpecialOperand map[string]arm64.SpecialOperand

// ARM64SpecialOperand returns the internal representation of a special operand.
func ARM64SpecialOperand(name string) arm64.SpecialOperand {
	if arm64SpecialOperand == nil {
		// Generate mapping when function is first called.
		arm64SpecialOperand = map[string]arm64.SpecialOperand{}
		for opd := arm64.SPOP_BEGIN; opd < arm64.SPOP_END; opd++ {
			arm64SpecialOperand[opd.String()] = opd
		}

		// Handle some special cases.
		specialMapping := map[string]arm64.SpecialOperand{
			// The internal representation of CS(CC) and HS(LO) are the same.
			"CS": arm64.SPOP_HS,
			"CC": arm64.SPOP_LO,
		}
		for s, opd := range specialMapping {
			arm64SpecialOperand[s] = opd
		}
	}
	if opd, ok := arm64SpecialOperand[name]; ok {
		return opd
	}
	return arm64.SPOP_END
}

// IsARM64ADR reports whether the op (as defined by an arm64.A* constant) is
// one of the comparison instructions that require special handling.
func IsARM64ADR(op obj.As) bool {
	switch op {
	case arm64.AADR, arm64.AADRP:
		return true
	}
	return false
}

// IsARM64CMP reports whether the op (as defined by an arm64.A* constant) is
// one of the comparison instructions that require special handling.
func IsARM64CMP(op obj.As) bool {
	switch op {
	case arm64.ACMN, arm64.ACMP, arm64.ATST,
		arm64.ACMNW, arm64.ACMPW, arm64.ATSTW,
		arm64.AFCMPS, arm64.AFCMPD,
		arm64.AFCMPES, arm64.AFCMPED:
		return true
	}
	return false
}

// IsARM64STLXR reports whether the op (as defined by an arm64.A*
// constant) is one of the STLXR-like instructions that require special
// handling.
func IsARM64STLXR(op obj.As) bool {
	switch op {
	case arm64.ASTLXRB, arm64.ASTLXRH, arm64.ASTLXRW, arm64.ASTLXR,
		arm64.ASTXRB, arm64.ASTXRH, arm64.ASTXRW, arm64.ASTXR,
		arm64.ASTXP, arm64.ASTXPW, arm64.ASTLXP, arm64.ASTLXPW:
		return true
	}
	// LDADDx/SWPx/CASx atomic instructions
	return arm64.IsAtomicInstruction(op)
}

// IsARM64TBL reports whether the op (as defined by an arm64.A*
// constant) is one of the TBL-like instructions and one of its
// inputs does not fit into prog.Reg, so require special handling.
func IsARM64TBL(op obj.As) bool {
	switch op {
	case arm64.AVTBL, arm64.AVTBX, arm64.AVMOVQ:
		return true
	}
	return false
}

// IsARM64CASP reports whether the op (as defined by an arm64.A*
// constant) is one of the CASP-like instructions, and its 2nd
// destination is a register pair that require special handling.
func IsARM64CASP(op obj.As) bool {
	switch op {
	case arm64.ACASPD, arm64.ACASPW:
		return true
	}
	return false
}

// ARM64Suffix handles the special suffix for the ARM64.
// It returns a boolean to indicate success; failure means
// cond was unrecognized.
func ARM64Suffix(prog *obj.Prog, cond string) bool {
	if cond == "" {
		return true
	}
	bits, ok := parseARM64Suffix(cond)
	if !ok {
		return false
	}
	prog.Scond = bits
	return true
}

// parseARM64Suffix parses the suffix attached to an ARM64 instruction.
// The input is a single string consisting of period-separated condition
// codes, such as ".P.W". An initial period is ignored.
func parseARM64Suffix(cond string) (uint8, bool) {
	if cond == "" {
		return 0, true
	}
	return parseARMCondition(cond, arm64LS, nil)
}

func arm64RegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "F":
		if 0 <= n && n <= 31 {
			return arm64.REG_F0 + n, true
		}
	case "R":
		if 0 <= n && n <= 30 { // not 31
			return arm64.REG_R0 + n, true
		}
	case "V":
		if 0 <= n && n <= 31 {
			return arm64.REG_V0 + n, true
		}
	}
	return 0, false
}

// ARM64RegisterShift constructs an ARM64 register with shift operation.
func ARM64RegisterShift(reg, op, count int16) (int64, error) {
	// the base register of shift operations must be general register.
	if reg > arm64.REG_R31 || reg < arm64.REG_R0 {
		return 0, errors.New("invalid register for shift operation")
	}
	return int64(reg&31)<<16 | int64(op)<<22 | int64(uint16(count)), nil
}

// ARM64RegisterArrangement constructs an ARM64 vector register arrangement.
func ARM64RegisterArrangement(reg int16, name, arng string) (int64, error) {
	var curQ, curSize uint16
	if name[0] != 'V' {
		return 0, errors.New("expect V0 through V31; found: " + name)
	}
	if reg < 0 {
		return 0, errors.New("invalid register number: " + name)
	}
	switch arng {
	case "B8":
		curSize = 0
		curQ = 0
	case "B16":
		curSize = 0
		curQ = 1
	case "H4":
		curSize = 1
		curQ = 0
	case "H8":
		curSize = 1
		curQ = 1
	case "S2":
		curSize = 2
		curQ = 0
	case "S4":
		curSize = 2
		curQ = 1
	case "D1":
		curSize = 3
		curQ = 0
	case "D2":
		curSize = 3
		curQ = 1
	default:
		return 0, errors.New("invalid arrangement in ARM64 register list")
	}
	return (int64(curQ) & 1 << 30) | (int64(curSize&3) << 10), nil
}

// ARM64RegisterListOffset generates offset encoding according to AArch64 specification.
func ARM64RegisterListOffset(firstReg, regCnt int, arrangement int64) (int64, error) {
	offset := int64(firstReg)
	switch regCnt {
	case 1:
		offset |= 0x7 << 12
	case 2:
		offset |= 0xa << 12
	case 3:
		offset |= 0x6 << 12
	case 4:
		offset |= 0x2 << 12
	default:
		return 0, errors.New("invalid register numbers in ARM64 register list")
	}
	offset |= arrangement
	// arm64 uses the 60th bit to differentiate from other archs
	// For more details, refer to: obj/arm64/list7.go
	offset |= 1 << 60
	return offset, nil
}
