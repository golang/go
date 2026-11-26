// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

// This file holds the operand model that SVE instructions need and that the
// NEON and base instruction sets do not: three register files, three
// arrangement qualifiers, the range spelling of a register list, a signed
// immediate, and the memory operands with a vector base or vector index.
//
// Nothing in this file is specific to one instruction. The decoder that
// produces these operands is not part of this file.

import (
	"fmt"
	"strings"
)

// The SVE scalable vector registers (Z), predicate registers (P) and
// predicate-as-counter registers (PN).
const (
	Z0 Reg = V31 + 1 + iota
	Z1
	Z2
	Z3
	Z4
	Z5
	Z6
	Z7
	Z8
	Z9
	Z10
	Z11
	Z12
	Z13
	Z14
	Z15
	Z16
	Z17
	Z18
	Z19
	Z20
	Z21
	Z22
	Z23
	Z24
	Z25
	Z26
	Z27
	Z28
	Z29
	Z30
	Z31

	P0
	P1
	P2
	P3
	P4
	P5
	P6
	P7
	P8
	P9
	P10
	P11
	P12
	P13
	P14
	P15

	PN0
	PN1
	PN2
	PN3
	PN4
	PN5
	PN6
	PN7
	PN8
	PN9
	PN10
	PN11
	PN12
	PN13
	PN14
	PN15
)

// sveRegString returns the spelling of r if it is one of the SVE register
// files, and "" otherwise.
func sveRegString(r Reg) string {
	switch {
	case Z0 <= r && r <= Z31:
		return fmt.Sprintf("Z%d", int(r-Z0))
	case P0 <= r && r <= P15:
		return fmt.Sprintf("P%d", int(r-P0))
	case PN0 <= r && r <= PN15:
		return fmt.Sprintf("PN%d", int(r-PN0))
	}
	return ""
}

// fileBase returns the first register of r's register file. Register lists
// wrap within a file, so a list of Z registers must expand from Z0 and not
// from V0.
func fileBase(r Reg) Reg {
	switch {
	case Z0 <= r && r <= Z31:
		return Z0
	case P0 <= r && r <= P15:
		return P0
	case PN0 <= r && r <= PN15:
		return PN0
	}
	return V0
}

// fileSize returns the number of registers in r's file, which is the modulus
// a register list wraps at.
func fileSize(r Reg) uint16 {
	if P0 <= r && r <= PN15 {
		return 16
	}
	return 32
}

// The SVE arrangement qualifiers. SVE also uses eight of the NEON
// arrangements: ".B", ".H", ".S" and ".D" for the elements of a Z register, and
// ".16B", ".8H", ".4S" and ".2D" for the SIMD&FP destination of quadword
// reductions such as ADDQV. ArrangementM and ArrangementZ are the merging and
// zeroing predication qualifiers, which ARM writes "P4/M" and "P4/Z".
// ArrangementQ is the plain quadword element, ".Q"; NEON only has
// Arrangement1Q. The values start at 64 to stay clear of the NEON ones.
const (
	ArrangementM Arrangement = 64 + iota
	ArrangementZ
	ArrangementQ
)

// sveArrangementString returns the spelling of a if it is an SVE-only
// arrangement, and "" otherwise.
func sveArrangementString(a Arrangement) string {
	switch a {
	case ArrangementM:
		return "/M"
	case ArrangementZ:
		return "/Z"
	case ArrangementQ:
		return ".Q"
	}
	return ""
}

// sveRangeBit is set in the cnt field of RegisterWithArrangement and
// RegisterWithArrangementAndIndex, the number of registers in a list, to mark a
// list that the Go assembler spells as a range ([Z22.B-Z23.B]) rather than
// enumerated ([Z22.B, Z23.B]). The count alone cannot tell the two apart at 2.
//
// The marker rides in cnt because both types are built with positional struct
// literals at hundreds of sites in decode.go, so adding a field would touch
// every one of them. NEON never counts above 4, so the top bit is free.
const sveRangeBit uint8 = 0x80

// listCount splits a cnt field into the register count and the range marker.
func listCount(cnt uint8) (uint8, bool) {
	return cnt &^ sveRangeBit, cnt&sveRangeBit != 0
}

// sveListString renders a list of cnt registers starting at r in GNU syntax.
//
// The range marker is deliberately not consulted: it distinguishes two
// spellings in Go syntax, and GNU syntax decides by count alone. A pair is
// enumerated, three or more registers are a range, and a group that wraps past
// the end of its file is enumerated again because "z30.b - z0.b" reads as an
// empty range. This is the stock NEON printer's rule and LLVM's, although GNU
// objdump prints a pair of SVE registers as a range.
func sveListString(r Reg, a Arrangement, cnt uint8) string {
	base, size := fileBase(r), fileSize(r)
	at := func(i int) string {
		cur := base + Reg((uint16(r)-uint16(base)+uint16(i))%size)
		return cur.String() + a.String()
	}
	if cnt <= 1 {
		return "{ " + at(0) + " }"
	}
	if cnt > 2 && uint16(cnt)+(uint16(r)-uint16(base))%size <= size {
		return "{ " + at(0) + " - " + at(int(cnt)-1) + " }"
	}
	parts := make([]string, cnt)
	for i := range parts {
		parts[i] = at(i)
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

// An ImmSigned is a signed immediate operand, printed "#-2". Imm64 always
// renders unsigned, and nothing in the base or NEON sets needs a signed
// immediate that is not a memory offset. SVE does (INDEX, ADDVL, RDVL, CPY).
type ImmSigned int64

func (ImmSigned) isArg() {}

func (i ImmSigned) String() string { return fmt.Sprintf("#%d", int64(i)) }

// A MemScaledVL is a memory reference whose immediate offset is counted in
// multiples of the vector length: [Xn, #imm, MUL VL]. The offset is always
// printed, including zero, because that addressing mode has no bare form.
type MemScaledVL struct {
	Base RegSP
	Imm  int32
}

func (MemScaledVL) isArg() {}

func (m MemScaledVL) String() string {
	return fmt.Sprintf("[%s, #%d, MUL VL]", m.Base, m.Imm)
}

// A MemVectorBase is a memory reference whose base is a vector register:
// [Zn.<T>{, #imm}]. MemImmediate.Base is a RegSP, which means "a general
// purpose register where 31 is the stack pointer".
type MemVectorBase struct {
	Base Reg
	Arr  Arrangement
	Imm  int32
}

func (MemVectorBase) isArg() {}

func (m MemVectorBase) String() string {
	if m.Imm == 0 {
		return fmt.Sprintf("[%s%s]", m.Base, m.Arr)
	}
	return fmt.Sprintf("[%s%s, #%d]", m.Base, m.Arr, m.Imm)
}

// A MemVectorIndex is a memory reference whose index is a vector register,
// with the arrangement on the index: [Xn, Zm.<T>{, <extend> {#amount}}].
// The base may itself be a vector (ADR), so it carries an optional
// arrangement too. Extend is zero when there is no extend or shift, and
// Amount is omitted from the spelling when it is zero.
//
// A general purpose base is spelled as RegSP, so register 31 is SP and not XZR.
type MemVectorIndex struct {
	Base    Reg
	BaseArr Arrangement
	Index   Reg
	IdxArr  Arrangement
	Extend  ExtShift
	Amount  uint8
}

func (MemVectorIndex) isArg() {}

func (m MemVectorIndex) String() string {
	base := m.Base.String() + m.BaseArr.String()
	if m.Base <= XZR {
		base = RegSP(m.Base).String()
	}
	s := fmt.Sprintf("[%s, %s%s", base, m.Index, m.IdxArr)
	if m.Extend != ExtShift(0) {
		s += ", " + m.Extend.String()
		if m.Amount != 0 {
			s += fmt.Sprintf(" #%d", m.Amount)
		}
	}
	return s + "]"
}

// A VLGroup is the vector group size of a multi-vector instruction, "vlx2" or
// "vlx4".
type VLGroup uint8

const (
	VLx2 VLGroup = 2
	VLx4 VLGroup = 4
)

func (VLGroup) isArg() {}

func (v VLGroup) String() string { return fmt.Sprintf("vlx%d", uint8(v)) }
