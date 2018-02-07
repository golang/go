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
}

func jumpArm64(word string) bool {
	return arm64Jump[word]
}

// IsARM64CMP reports whether the op (as defined by an arm.A* constant) is
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
		arm64.ASTXRB, arm64.ASTXRH, arm64.ASTXRW, arm64.ASTXR:
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
	bits, ok := ParseARM64Suffix(cond)
	if !ok {
		return false
	}
	prog.Scond = bits
	return true
}

// ParseARM64Suffix parses the suffix attached to an ARM64 instruction.
// The input is a single string consisting of period-separated condition
// codes, such as ".P.W". An initial period is ignored.
func ParseARM64Suffix(cond string) (uint8, bool) {
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

// ARM64RegisterExtension parses an ARM64 register with extension or arrangment.
func ARM64RegisterExtension(a *obj.Addr, ext string, reg, num int16, isAmount, isIndex bool) error {
	rm := uint32(reg)
	if isAmount {
		if num < 0 || num > 7 {
			return errors.New("shift amount out of range")
		}
	}
	switch ext {
	case "UXTB":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_UXTB + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (uint32(num) << 10))
	case "UXTH":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_UXTH + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (1 << 13) | (uint32(num) << 10))
	case "UXTW":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_UXTW + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (2 << 13) | (uint32(num) << 10))
	case "UXTX":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_UXTX + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (3 << 13) | (uint32(num) << 10))
	case "SXTB":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_SXTB + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (4 << 13) | (uint32(num) << 10))
	case "SXTH":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_SXTH + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (5 << 13) | (uint32(num) << 10))
	case "SXTW":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_SXTW + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (6 << 13) | (uint32(num) << 10))
	case "SXTX":
		if !isAmount {
			return errors.New("invalid register extension")
		}
		a.Reg = arm64.REG_SXTX + (reg & 31) + int16(num<<5)
		a.Offset = int64(((rm & 31) << 16) | (7 << 13) | (uint32(num) << 10))
	case "B8":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_8B & 15) << 5)
	case "B16":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_16B & 15) << 5)
	case "H4":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_4H & 15) << 5)
	case "H8":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_8H & 15) << 5)
	case "S2":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_2S & 15) << 5)
	case "S4":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_4S & 15) << 5)
	case "D2":
		a.Reg = arm64.REG_ARNG + (reg & 31) + ((arm64.ARNG_2D & 15) << 5)
	case "B":
		if !isIndex {
			return nil
		}
		a.Reg = arm64.REG_ELEM + (reg & 31) + ((arm64.ARNG_B & 15) << 5)
		a.Index = num
	case "H":
		if !isIndex {
			return nil
		}
		a.Reg = arm64.REG_ELEM + (reg & 31) + ((arm64.ARNG_H & 15) << 5)
		a.Index = num
	case "S":
		if !isIndex {
			return nil
		}
		a.Reg = arm64.REG_ELEM + (reg & 31) + ((arm64.ARNG_S & 15) << 5)
		a.Index = num
	case "D":
		if !isIndex {
			return nil
		}
		a.Reg = arm64.REG_ELEM + (reg & 31) + ((arm64.ARNG_D & 15) << 5)
		a.Index = num
	default:
		return errors.New("unsupported register extension type: " + ext)
	}
	a.Type = obj.TYPE_REG
	return nil
}

// ARM64RegisterArrangement parses an ARM64 vector register arrangment.
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
		curSize = 1
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
