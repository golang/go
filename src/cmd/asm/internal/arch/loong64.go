// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the
// Loong64 (LoongArch64) instruction set, to minimize its interaction
// with the core of the assembler.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/loong64"
	"errors"
	"fmt"
)

func jumpLoong64(word string) bool {
	switch word {
	case "BEQ", "BFPF", "BFPT", "BLTZ", "BGEZ", "BLEZ", "BGTZ", "BLT", "BLTU", "JIRL", "BNE", "BGE", "BGEU", "JMP", "JAL", "CALL":
		return true
	}
	return false
}

// IsLoong64MUL reports whether the op (as defined by an loong64.A* constant) is
// one of the MUL/DIV/REM instructions that require special handling.
func IsLoong64MUL(op obj.As) bool {
	switch op {
	case loong64.AMUL, loong64.AMULU, loong64.AMULV, loong64.AMULVU,
		loong64.ADIV, loong64.ADIVU, loong64.ADIVV, loong64.ADIVVU,
		loong64.AREM, loong64.AREMU, loong64.AREMV, loong64.AREMVU:
		return true
	}
	return false
}

// IsLoong64RDTIME reports whether the op (as defined by an loong64.A*
// constant) is one of the RDTIMELW/RDTIMEHW/RDTIMED instructions that
// require special handling.
func IsLoong64RDTIME(op obj.As) bool {
	switch op {
	case loong64.ARDTIMELW, loong64.ARDTIMEHW, loong64.ARDTIMED:
		return true
	}
	return false
}

func IsLoong64PRELD(op obj.As) bool {
	switch op {
	case loong64.APRELD, loong64.APRELDX:
		return true
	}
	return false
}

func IsLoong64AMO(op obj.As) bool {
	return loong64.IsAtomicInst(op)
}

var loong64ElemExtMap = map[string]int16{
	"B":  loong64.ARNG_B,
	"H":  loong64.ARNG_H,
	"W":  loong64.ARNG_W,
	"V":  loong64.ARNG_V,
	"BU": loong64.ARNG_BU,
	"HU": loong64.ARNG_HU,
	"WU": loong64.ARNG_WU,
	"VU": loong64.ARNG_VU,
}

var loong64LsxArngExtMap = map[string]int16{
	"B16": loong64.ARNG_16B,
	"H8":  loong64.ARNG_8H,
	"W4":  loong64.ARNG_4W,
	"V2":  loong64.ARNG_2V,
}

var loong64LasxArngExtMap = map[string]int16{
	"B32": loong64.ARNG_32B,
	"H16": loong64.ARNG_16H,
	"W8":  loong64.ARNG_8W,
	"V4":  loong64.ARNG_4V,
	"Q2":  loong64.ARNG_2Q,
}

// Loong64RegisterExtension constructs an Loong64 register with extension or arrangement.
func Loong64RegisterExtension(a *obj.Addr, ext string, reg, num int16, isAmount, isIndex bool) error {
	var ok bool
	var arng_type int16
	var simd_type int16

	switch {
	case reg >= loong64.REG_V0 && reg <= loong64.REG_V31:
		simd_type = loong64.LSX
	case reg >= loong64.REG_X0 && reg <= loong64.REG_X31:
		simd_type = loong64.LASX
	default:
		return errors.New("Loong64 extension: invalid LSX/LASX register: " + fmt.Sprintf("%d", reg))
	}

	if isIndex {
		arng_type, ok = loong64ElemExtMap[ext]
		if !ok {
			return errors.New("Loong64 extension: invalid LSX/LASX arrangement type: " + ext)
		}

		a.Reg = loong64.REG_ELEM
		a.Reg += ((reg & loong64.EXT_REG_MASK) << loong64.EXT_REG_SHIFT)
		a.Reg += ((arng_type & loong64.EXT_TYPE_MASK) << loong64.EXT_TYPE_SHIFT)
		a.Reg += ((simd_type & loong64.EXT_SIMDTYPE_MASK) << loong64.EXT_SIMDTYPE_SHIFT)
		a.Index = num
	} else {
		switch simd_type {
		case loong64.LSX:
			arng_type, ok = loong64LsxArngExtMap[ext]
			if !ok {
				return errors.New("Loong64 extension: invalid LSX arrangement type: " + ext)
			}

		case loong64.LASX:
			arng_type, ok = loong64LasxArngExtMap[ext]
			if !ok {
				return errors.New("Loong64 extension: invalid LASX arrangement type: " + ext)
			}
		}

		a.Reg = loong64.REG_ARNG
		a.Reg += ((reg & loong64.EXT_REG_MASK) << loong64.EXT_REG_SHIFT)
		a.Reg += ((arng_type & loong64.EXT_TYPE_MASK) << loong64.EXT_TYPE_SHIFT)
		a.Reg += ((simd_type & loong64.EXT_SIMDTYPE_MASK) << loong64.EXT_SIMDTYPE_SHIFT)
	}

	return nil
}

func loong64RegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "F":
		if 0 <= n && n <= 31 {
			return loong64.REG_F0 + n, true
		}
	case "FCSR":
		if 0 <= n && n <= 31 {
			return loong64.REG_FCSR0 + n, true
		}
	case "FCC":
		if 0 <= n && n <= 31 {
			return loong64.REG_FCC0 + n, true
		}
	case "R":
		if 0 <= n && n <= 31 {
			return loong64.REG_R0 + n, true
		}
	case "V":
		if 0 <= n && n <= 31 {
			return loong64.REG_V0 + n, true
		}
	case "X":
		if 0 <= n && n <= 31 {
			return loong64.REG_X0 + n, true
		}
	}
	return 0, false
}
