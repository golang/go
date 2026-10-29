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
)

func jumpLoong64(word string) bool {
	switch word {
	case "BEQ", "BFPF", "BFPT", "BLTZ", "BGEZ", "BLEZ", "BGTZ", "BLT", "BLTU", "BNE", "BGE", "BGEU", "JMP", "JAL", "CALL":
		return true
	}
	return false
}

func IsLoong64JIRL(op obj.As) bool {
	return op == loong64.AJAL
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
