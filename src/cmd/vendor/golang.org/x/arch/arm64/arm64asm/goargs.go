// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

// This file prints SVE instructions in Go assembler syntax. sveargs.go holds
// the operands and their GNU syntax.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// sveGoSyntax renders an SVE instruction in Go assembler syntax.
//
// Two things differ from the GNU spelling. The mnemonic is the Go assembler's,
// which is data (sveOpGo). The operands are in the reverse order except for
// stores, whose ARM mnemonics all start with ST.
func sveGoSyntax(inst Inst) string {
	var args []string
	for _, a := range inst.Args {
		if a == nil {
			break
		}
		args = append(args, sveGoArg(a))
	}
	if !strings.HasPrefix(inst.Op.String(), "ST") {
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}
	}
	op := sveOpGo[inst.Op-sveOpBase]
	if len(args) == 0 {
		return op
	}
	return op + " " + strings.Join(args, ", ")
}

// goArrangement converts an ARM arrangement spelling to the Go assembler's: ARM
// writes the lane count first ("4S"), Go writes it last ("S4"). SVE's own
// arrangements are single letters and pass through.
func goArrangement(a Arrangement) string {
	s := a.String()
	if s == "" {
		return s
	}
	if s[0] == '/' {
		return "." + s[1:]
	}
	i := 1
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	return "." + s[i:] + s[1:i]
}

// goReg spells a register in Go assembler syntax.
//
// The Go assembler reserves R18 and R28 and names them R18_PLATFORM and g, and
// rejects the numeric spellings ("illegal or missing addressing mode for symbol
// R18"). plan9gpr, which the other instructions use, still writes the numeric
// forms, so SVE spells them itself to keep its output assemblable. The zero
// register is spelled ZR by plan9gpr, never R31.
func goReg(r Reg) string {
	if isGPR(r) {
		switch uint16(r) & 31 {
		case 18:
			return "R18_PLATFORM"
		case 28:
			return "g"
		}
		return plan9gpr(r)
	}
	if B0 <= r && r <= Q31 {
		// The Go assembler writes every SIMD&FP scalar as Vn regardless of width,
		// which is what the SVE reductions' testdata pins ("ZCLASTAD Z9.D, V10").
		return fmt.Sprintf("V%d", uint16(r)&31)
	}
	return r.String()
}

// sveGoArg renders one Arg in Go syntax. One case per Arg type the SVE tables
// produce; the memory cases all share Go's "<index-or-offset>(<base>)" shape.
func sveGoArg(arg Arg) string {
	switch a := arg.(type) {
	case Reg:
		return goReg(a)

	case RegSP:
		if uint16(a)&31 == 31 {
			return "RSP"
		}
		return goReg(Reg(a))

	case RegisterWithArrangement:
		n, rng := listCount(a.cnt)
		s := goReg(a.r) + goArrangement(a.a)
		if n == 0 {
			return s
		}
		base, size := fileBase(a.r), fileSize(a.r)
		at := func(i int) string {
			cur := base + Reg((uint16(a.r)-uint16(base)+uint16(i))%size)
			return goReg(cur) + goArrangement(a.a)
		}
		if rng {
			return "[" + at(0) + "-" + at(int(n)-1) + "]"
		}
		parts := make([]string, n)
		for i := range parts {
			parts[i] = at(i)
		}
		return "[" + strings.Join(parts, ", ") + "]"

	case RegisterWithArrangementAndIndex:
		return fmt.Sprintf("%s%s[%d]", goReg(a.r), goArrangement(a.a), a.index)

	case Imm64:
		if !a.Decimal {
			return fmt.Sprintf("$%d", a.Imm)
		}
		return fmt.Sprintf("$%d", int64(a.Imm))

	case ImmSigned:
		return fmt.Sprintf("$%d", int64(a))

	case sveZeroFP:
		return "$(0.0)"

	case Imm_fp:
		// Go spells these "$(1.0)", "$(0.5)". Formatting the value is right;
		// editing Imm_fp.String's "#1.000000000000000000e+00" is not, and produces
		// "$(1.000000000000000000e+.0)".
		v := math.Ldexp(float64(16+int64(a.pre))/16, int(a.exp))
		if a.s != 0 {
			v = -v
		}
		s := strconv.FormatFloat(v, 'g', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return "$(" + s + ")"

	case Imm_prfop:
		return a.String()

	case VLGroup:
		return fmt.Sprintf("VLx%d", uint8(a))

	case MemImmediate:
		if a.imm == 0 {
			return fmt.Sprintf("(%s)", sveGoArg(a.Base))
		}
		return fmt.Sprintf("%d(%s)", a.imm, sveGoArg(a.Base))

	case MemScaledVL:
		// The VL multiplier is spelled even when it is zero. cmd/asm rejects the
		// bare "(R7)" for this addressing mode -- "illegal combination from SVE" --
		// because the MUL VL scaling is part of the mode rather than an optional
		// offset, and LLVM prints "#0, mul vl" for the same reason.
		if a.Imm < 0 {
			return fmt.Sprintf("(-VL*%d)(%s)", -a.Imm, sveGoArg(a.Base))
		}
		return fmt.Sprintf("(VL*%d)(%s)", a.Imm, sveGoArg(a.Base))

	case MemVectorBase:
		if a.Imm == 0 {
			return fmt.Sprintf("(%s%s)", goReg(a.Base), goArrangement(a.Arr))
		}
		return fmt.Sprintf("%d(%s%s)", a.Imm, goReg(a.Base), goArrangement(a.Arr))

	case MemExtend:
		return fmt.Sprintf("(%s)(%s)", goIndex(a.Index, 0, a.Extend, a.Amount),
			sveGoArg(a.Base))

	case MemVectorIndex:
		return fmt.Sprintf("(%s)(%s)",
			goIndex(a.Index, a.IdxArr, a.Extend, a.Amount),
			goMemBase(a.Base, a.BaseArr))
	}
	return strings.ToUpper(arg.String())
}

// goMemBase spells a memory base. A general-purpose base is "a GPR where 31 is
// the stack pointer", so it goes through the RegSP spelling; a vector base keeps
// its arrangement.
func goMemBase(r Reg, arr Arrangement) string {
	if isGPR(r) {
		return sveGoArg(RegSP(r))
	}
	return goReg(r) + goArrangement(arr)
}

// goIndex spells a memory index in Go syntax: the register, then its arrangement
// and extend as a dotted suffix, then the shift as "<<n". Go writes an extended
// GPR index the same way the base+extend forms elsewhere in plan9x.go do.
func goIndex(r Reg, arr Arrangement, ext ExtShift, amt uint8) string {
	s := goReg(r) + goArrangement(arr)
	if ext != ExtShift(0) && ext != lsl {
		s += "." + ext.String()
	}
	if amt != 0 {
		s += fmt.Sprintf("<<%d", amt)
	}
	return s
}
