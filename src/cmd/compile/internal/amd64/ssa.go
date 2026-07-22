// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"fmt"
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"internal/abi"
	"internal/buildcfg"
)

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *ssagen.State, b *ssacore.Block) {
	flive := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		flive = c.Type.IsFlags() || flive
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && (v.Op == ssaop.OpAMD64MOVLconst || v.Op == ssaop.OpAMD64MOVQconst) {
			// The "mark" is any non-nil Aux value.
			v.Aux = ssacore.AuxMark
		}
		if v.Type.IsFlags() {
			flive = false
		}
		for _, a := range v.Args {
			if a.Type.IsFlags() {
				flive = true
			}
		}
	}
}

func isGPReg(r int16) bool {
	return x86.REG_AL <= r && r <= x86.REG_R15
}

func isFPReg(r int16) bool {
	return x86.REG_X0 <= r && r <= x86.REG_Z31
}

func isKReg(r int16) bool {
	return x86.REG_K0 <= r && r <= x86.REG_K7
}

func isLowFPReg(r int16) bool {
	return x86.REG_X0 <= r && r <= x86.REG_X15
}

func isHighFPReg(r int16) bool {
	return x86.REG_X16 <= r && r <= x86.REG_X31 || x86.REG_Y16 <= r && r <= x86.REG_Y31 || x86.REG_Z16 <= r && r <= x86.REG_Z31
}

// loadByRegWidth returns the load instruction of the given register of a given width.
func loadByRegWidth(r int16, width int64) obj.As {
	// Avoid partial register write for GPR
	if !isFPReg(r) && !isKReg(r) {
		switch width {
		case 1:
			return x86.AMOVBLZX
		case 2:
			return x86.AMOVWLZX
		}
	}
	// Otherwise, there's no difference between load and store opcodes.
	return storeByRegWidth(r, width)
}

// storeByRegWidth returns the store instruction of the given register of a given width.
// It's also used for loading const to a reg.
func storeByRegWidth(r int16, width int64) obj.As {
	if isHighFPReg(r) {
		// High registers require AVX512 instruction
		return x86.AVMOVDQU64
	}
	if isFPReg(r) {
		switch width {
		case 4:
			return x86.AMOVSS
		case 8:
			return x86.AMOVSD
		case 16:
			// int128s are in SSE registers
			return x86.AMOVUPS
		case 32:
			return x86.AVMOVDQU
		case 64:
			return x86.AVMOVDQU64
		}
	}
	if isKReg(r) {
		return x86.AKMOVQ
	}
	// gp
	switch width {
	case 1:
		return x86.AMOVB
	case 2:
		return x86.AMOVW
	case 4:
		return x86.AMOVL
	case 8:
		return x86.AMOVQ
	}
	panic(fmt.Sprintf("bad store reg=%v, width=%d", r, width))
}

// moveByRegsWidth returns the reg->reg move instruction of the given dest/src registers of a given width.
func moveByRegsWidth(dest, src int16, width int64) obj.As {
	// fp -> fp
	if isFPReg(dest) && isFPReg(src) {
		if isHighFPReg(src) || isHighFPReg(dest) {
			// High registers require AVX512 instruction
			return x86.AVMOVDQU64
		}
		// Moving the whole sse2 register is faster
		// than moving just the correct low portion of it.
		// There is no xmm->xmm move with 1 byte opcode,
		// so use movups, which has 2 byte opcode.
		if width <= 16 {
			return x86.AMOVUPS
		}
		if width <= 32 {
			return x86.AVMOVDQU
		}
		return x86.AVMOVDQU64
	}
	// k -> gp, gp -> k, k -> k
	if isKReg(dest) || isKReg(src) {
		if isFPReg(dest) || isFPReg(src) {
			panic(fmt.Sprintf("bad move, src=%v, dest=%v, width=%d", src, dest, width))
		}
		return x86.AKMOVQ
	}
	// gp -> fp, fp -> gp, gp -> gp
	switch width {
	case 1:
		// Avoids partial register write
		return x86.AMOVL
	case 2:
		return x86.AMOVL
	case 4:
		return x86.AMOVL
	case 8:
		return x86.AMOVQ
	case 16:
		// int128s are in SSE registers
		return x86.AMOVUPS
	case 32:
		return x86.AVMOVDQU
	case 64:
		return x86.AVMOVDQU64
	}
	panic(fmt.Sprintf("bad move, src=%v, dest=%v, width=%d", src, dest, width))
}

// opregreg emits instructions for
//
//	dest := dest(To) op src(From)
//
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregreg(s *ssagen.State, op obj.As, dest, src int16) *obj.Prog {
	p := s.Prog(op)
	p.From.Type = obj.TYPE_REG
	p.To.Type = obj.TYPE_REG
	p.To.Reg = dest
	p.From.Reg = src
	return p
}

// memIdx fills out a as an indexed memory reference for v.
// It assumes that the base register and the index register
// are v.Args[0].Reg() and v.Args[1].Reg(), respectively.
// The caller must still use gc.AddAux/gc.AddAux2 to handle v.Aux as necessary.
func memIdx(a *obj.Addr, v *ssacore.Value) {
	r, i := v.Args[0].Reg(), v.Args[1].Reg()
	a.Type = obj.TYPE_MEM
	a.Scale = v.Op.Scale()
	if a.Scale == 1 && i == x86.REG_SP {
		r, i = i, r
	}
	a.Reg = r
	a.Index = i
}

func getgFromTLS(s *ssagen.State, r int16) {
	// See the comments in cmd/internal/obj/x86/obj6.go
	// near CanUse1InsnTLS for a detailed explanation of these instructions.
	if x86.CanUse1InsnTLS(base.Ctxt) {
		// MOVQ (TLS), r
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = x86.REG_TLS
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	} else {
		// MOVQ TLS, r
		// MOVQ (r)(TLS*1), r
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_TLS
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		q := s.Prog(x86.AMOVQ)
		q.From.Type = obj.TYPE_MEM
		q.From.Reg = r
		q.From.Index = x86.REG_TLS
		q.From.Scale = 1
		q.To.Type = obj.TYPE_REG
		q.To.Reg = r
	}
}

func ssaGenValue(s *ssagen.State, v *ssacore.Value) {
	switch v.Op {
	case ssaop.OpAMD64VFMADD231SD, ssaop.OpAMD64VFMADD231SS, ssaop.OpAMD64VFMSUB231SD, ssaop.OpAMD64VFMSUB231SS, ssaop.OpAMD64VFNMADD231SD, ssaop.OpAMD64VFNMADD231SS:
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: v.Args[2].Reg()}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: v.Reg()}
		p.AddRestSourceReg(v.Args[1].Reg())
	case ssaop.OpAMD64ADDQ, ssaop.OpAMD64ADDL:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		switch {
		case r == r1:
			p := s.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r2
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		case r == r2:
			p := s.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r1
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		default:
			var asm obj.As
			if v.Op == ssaop.OpAMD64ADDQ {
				asm = x86.ALEAQ
			} else {
				asm = x86.ALEAL
			}
			p := s.Prog(asm)
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = r1
			p.From.Scale = 1
			p.From.Index = r2
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	// 2-address opcode arithmetic
	case ssaop.OpAMD64SUBQ, ssaop.OpAMD64SUBL,
		ssaop.OpAMD64MULQ, ssaop.OpAMD64MULL,
		ssaop.OpAMD64ANDQ, ssaop.OpAMD64ANDL,
		ssaop.OpAMD64ORQ, ssaop.OpAMD64ORL,
		ssaop.OpAMD64XORQ, ssaop.OpAMD64XORL,
		ssaop.OpAMD64SHLQ, ssaop.OpAMD64SHLL,
		ssaop.OpAMD64SHRQ, ssaop.OpAMD64SHRL, ssaop.OpAMD64SHRW, ssaop.OpAMD64SHRB,
		ssaop.OpAMD64SARQ, ssaop.OpAMD64SARL, ssaop.OpAMD64SARW, ssaop.OpAMD64SARB,
		ssaop.OpAMD64ROLQ, ssaop.OpAMD64ROLL, ssaop.OpAMD64ROLW, ssaop.OpAMD64ROLB,
		ssaop.OpAMD64RORQ, ssaop.OpAMD64RORL, ssaop.OpAMD64RORW, ssaop.OpAMD64RORB,
		ssaop.OpAMD64ADDSS, ssaop.OpAMD64ADDSD, ssaop.OpAMD64SUBSS, ssaop.OpAMD64SUBSD,
		ssaop.OpAMD64MULSS, ssaop.OpAMD64MULSD, ssaop.OpAMD64DIVSS, ssaop.OpAMD64DIVSD,
		ssaop.OpAMD64MINSS, ssaop.OpAMD64MINSD,
		ssaop.OpAMD64MAXSS, ssaop.OpAMD64MAXSD,
		ssaop.OpAMD64POR, ssaop.OpAMD64PXOR,
		ssaop.OpAMD64BTSL, ssaop.OpAMD64BTSQ,
		ssaop.OpAMD64BTCL, ssaop.OpAMD64BTCQ,
		ssaop.OpAMD64BTRL, ssaop.OpAMD64BTRQ,
		ssaop.OpAMD64PCMPEQB, ssaop.OpAMD64PSIGNB,
		ssaop.OpAMD64PUNPCKLBW:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())

	case ssaop.OpAMD64PSHUFLW:
		p := s.Prog(v.Op.Asm())
		imm := v.AuxInt
		if imm < 0 || imm > 255 {
			v.Fatalf("Invalid source selection immediate")
		}
		p.From.Offset = imm
		p.From.Type = obj.TYPE_CONST
		p.AddRestSourceReg(v.Args[0].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64PSHUFBbroadcast:
		// PSHUFB with a control mask of zero copies byte 0 to all
		// bytes in the register.
		//
		// X15 is always zero with ABIInternal.
		if s.ABI != obj.ABIInternal {
			// zero X15 manually
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
		}

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.From.Reg = x86.REG_X15

	case ssaop.OpAMD64BLSIQ, ssaop.OpAMD64BLSIL,
		ssaop.OpAMD64BLSMSKQ, ssaop.OpAMD64BLSMSKL,
		ssaop.OpAMD64BLSRQ, ssaop.OpAMD64BLSRL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		switch v.Op {
		case ssaop.OpAMD64BLSRQ, ssaop.OpAMD64BLSRL:
			p.To.Reg = v.Reg0()
		default:
			p.To.Reg = v.Reg()
		}

	case ssaop.OpAMD64ANDNQ, ssaop.OpAMD64ANDNL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.AddRestSourceReg(v.Args[1].Reg())

	case ssaop.OpAMD64SARXL, ssaop.OpAMD64SARXQ,
		ssaop.OpAMD64SHLXL, ssaop.OpAMD64SHLXQ,
		ssaop.OpAMD64SHRXL, ssaop.OpAMD64SHRXQ:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		p.AddRestSourceReg(v.Args[0].Reg())

	case ssaop.OpAMD64SHLXLload, ssaop.OpAMD64SHLXQload,
		ssaop.OpAMD64SHRXLload, ssaop.OpAMD64SHRXQload,
		ssaop.OpAMD64SARXLload, ssaop.OpAMD64SARXQload:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[0].Reg()}
		ssagen.AddAux(&m, v)
		p.AddRestSource(m)

	case ssaop.OpAMD64SHLXLloadidx1, ssaop.OpAMD64SHLXLloadidx4, ssaop.OpAMD64SHLXLloadidx8,
		ssaop.OpAMD64SHRXLloadidx1, ssaop.OpAMD64SHRXLloadidx4, ssaop.OpAMD64SHRXLloadidx8,
		ssaop.OpAMD64SARXLloadidx1, ssaop.OpAMD64SARXLloadidx4, ssaop.OpAMD64SARXLloadidx8,
		ssaop.OpAMD64SHLXQloadidx1, ssaop.OpAMD64SHLXQloadidx8,
		ssaop.OpAMD64SHRXQloadidx1, ssaop.OpAMD64SHRXQloadidx8,
		ssaop.OpAMD64SARXQloadidx1, ssaop.OpAMD64SARXQloadidx8:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[2].Reg())
		m := obj.Addr{Type: obj.TYPE_MEM}
		memIdx(&m, v)
		ssagen.AddAux(&m, v)
		p.AddRestSource(m)

	case ssaop.OpAMD64DIVQU, ssaop.OpAMD64DIVLU, ssaop.OpAMD64DIVWU:
		// Arg[0] (the dividend) is in AX.
		// Arg[1] (the divisor) can be in any other register.
		// Result[0] (the quotient) is in AX.
		// Result[1] (the remainder) is in DX.
		r := v.Args[1].Reg()

		// Zero extend dividend.
		opregreg(s, x86.AXORL, x86.REG_DX, x86.REG_DX)

		// Issue divide.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r

	case ssaop.OpAMD64DIVQ, ssaop.OpAMD64DIVL, ssaop.OpAMD64DIVW:
		// Arg[0] (the dividend) is in AX.
		// Arg[1] (the divisor) can be in any other register.
		// Result[0] (the quotient) is in AX.
		// Result[1] (the remainder) is in DX.
		r := v.Args[1].Reg()

		var opCMP, opNEG, opSXD obj.As
		switch v.Op {
		case ssaop.OpAMD64DIVQ:
			opCMP, opNEG, opSXD = x86.ACMPQ, x86.ANEGQ, x86.ACQO
		case ssaop.OpAMD64DIVL:
			opCMP, opNEG, opSXD = x86.ACMPL, x86.ANEGL, x86.ACDQ
		case ssaop.OpAMD64DIVW:
			opCMP, opNEG, opSXD = x86.ACMPW, x86.ANEGW, x86.ACWD
		}

		// CPU faults upon signed overflow, which occurs when the most
		// negative int is divided by -1. Handle divide by -1 as a special case.
		var j1, j2 *obj.Prog
		if ssacore.DivisionNeedsFixUp(v) {
			c := s.Prog(opCMP)
			c.From.Type = obj.TYPE_REG
			c.From.Reg = r
			c.To.Type = obj.TYPE_CONST
			c.To.Offset = -1

			// Divisor is not -1, proceed with normal division.
			j1 = s.Prog(x86.AJNE)
			j1.To.Type = obj.TYPE_BRANCH

			// Divisor is -1, manually compute quotient and remainder via fixup code.
			// n / -1 = -n
			n1 := s.Prog(opNEG)
			n1.To.Type = obj.TYPE_REG
			n1.To.Reg = x86.REG_AX

			// n % -1 == 0
			opregreg(s, x86.AXORL, x86.REG_DX, x86.REG_DX)

			// TODO(khr): issue only the -1 fixup code we need.
			// For instance, if only the quotient is used, no point in zeroing the remainder.

			// Skip over normal division.
			j2 = s.Prog(obj.AJMP)
			j2.To.Type = obj.TYPE_BRANCH
		}

		// Sign extend dividend and perform division.
		p := s.Prog(opSXD)
		if j1 != nil {
			j1.To.SetTarget(p)
		}
		p = s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r

		if j2 != nil {
			j2.To.SetTarget(s.Pc())
		}

	case ssaop.OpAMD64HMULQ, ssaop.OpAMD64HMULL, ssaop.OpAMD64HMULQU, ssaop.OpAMD64HMULLU:
		// the frontend rewrites constant division by 8/16/32 bit integers into
		// HMUL by a constant
		// SSA rewrites generate the 64 bit versions

		// Arg[0] is already in AX as it's the only register we allow
		// and DX is the only output we care about (the high bits)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

		// IMULB puts the high portion in AH instead of DL,
		// so move it to DL for consistency
		if v.Type.Size() == 1 {
			m := s.Prog(x86.AMOVB)
			m.From.Type = obj.TYPE_REG
			m.From.Reg = x86.REG_AH
			m.To.Type = obj.TYPE_REG
			m.To.Reg = x86.REG_DX
		}

	case ssaop.OpAMD64MULQU, ssaop.OpAMD64MULLU:
		// Arg[0] is already in AX as it's the only register we allow
		// results lo in AX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssaop.OpAMD64MULQU2:
		// Arg[0] is already in AX as it's the only register we allow
		// results hi in DX, lo in AX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssaop.OpAMD64MULXQ:
		// Arg[0] is already in DX (the implicit operand); Arg[1] is any GP/mem.
		// SSA outputs are (hi, lo) -> Reg0()=hi, Reg1()=lo.
		// Go assembler syntax: MULXQ src, lo, hi (encodes vvvv=lo, reg=hi).
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.AddRestSourceReg(v.Reg1())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpAMD64DIVQU2:
		// Arg[0], Arg[1] are already in Dx, AX, as they're the only registers we allow
		// results q in AX, r in DX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()

	case ssaop.OpAMD64AVGQU:
		// compute (x+y)/2 unsigned.
		// Do a 64-bit add, the overflow goes into the carry.
		// Shift right once and pull the carry back into the 63rd bit.
		p := s.Prog(x86.AADDQ)
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.From.Reg = v.Args[1].Reg()
		p = s.Prog(x86.ARCRQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64ADDQcarry, ssaop.OpAMD64ADCQ:
		r := v.Reg0()
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		switch r {
		case r0:
			p := s.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r1
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		case r1:
			p := s.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		default:
			v.Fatalf("output not in same register as an input %s", v.LongString())
		}

	case ssaop.OpAMD64SUBQborrow, ssaop.OpAMD64SBBQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpAMD64ADDQconstcarry, ssaop.OpAMD64ADCQconst, ssaop.OpAMD64SUBQconstborrow, ssaop.OpAMD64SBBQconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpAMD64ADDQconst, ssaop.OpAMD64ADDLconst:
		r := v.Reg()
		a := v.Args[0].Reg()
		if r == a {
			switch v.AuxInt {
			case 1:
				var asm obj.As
				// Software optimization manual recommends add $1,reg.
				// But inc/dec is 1 byte smaller. ICC always uses inc
				// Clang/GCC choose depending on flags, but prefer add.
				// Experiments show that inc/dec is both a little faster
				// and make a binary a little smaller.
				if v.Op == ssaop.OpAMD64ADDQconst {
					asm = x86.AINCQ
				} else {
					asm = x86.AINCL
				}
				p := s.Prog(asm)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return
			case -1:
				var asm obj.As
				if v.Op == ssaop.OpAMD64ADDQconst {
					asm = x86.ADECQ
				} else {
					asm = x86.ADECL
				}
				p := s.Prog(asm)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return
			case 0x80:
				// 'SUBQ $-0x80, r' is shorter to encode than
				// and functionally equivalent to 'ADDQ $0x80, r'.
				asm := x86.ASUBL
				if v.Op == ssaop.OpAMD64ADDQconst {
					asm = x86.ASUBQ
				}
				p := s.Prog(asm)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = -0x80
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return

			}
			p := s.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = v.AuxInt
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			return
		}
		var asm obj.As
		if v.Op == ssaop.OpAMD64ADDQconst {
			asm = x86.ALEAQ
		} else {
			asm = x86.ALEAL
		}
		p := s.Prog(asm)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = a
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssaop.OpAMD64CMOVQEQ, ssaop.OpAMD64CMOVLEQ, ssaop.OpAMD64CMOVWEQ,
		ssaop.OpAMD64CMOVQLT, ssaop.OpAMD64CMOVLLT, ssaop.OpAMD64CMOVWLT,
		ssaop.OpAMD64CMOVQNE, ssaop.OpAMD64CMOVLNE, ssaop.OpAMD64CMOVWNE,
		ssaop.OpAMD64CMOVQGT, ssaop.OpAMD64CMOVLGT, ssaop.OpAMD64CMOVWGT,
		ssaop.OpAMD64CMOVQLE, ssaop.OpAMD64CMOVLLE, ssaop.OpAMD64CMOVWLE,
		ssaop.OpAMD64CMOVQGE, ssaop.OpAMD64CMOVLGE, ssaop.OpAMD64CMOVWGE,
		ssaop.OpAMD64CMOVQHI, ssaop.OpAMD64CMOVLHI, ssaop.OpAMD64CMOVWHI,
		ssaop.OpAMD64CMOVQLS, ssaop.OpAMD64CMOVLLS, ssaop.OpAMD64CMOVWLS,
		ssaop.OpAMD64CMOVQCC, ssaop.OpAMD64CMOVLCC, ssaop.OpAMD64CMOVWCC,
		ssaop.OpAMD64CMOVQCS, ssaop.OpAMD64CMOVLCS, ssaop.OpAMD64CMOVWCS,
		ssaop.OpAMD64CMOVQGTF, ssaop.OpAMD64CMOVLGTF, ssaop.OpAMD64CMOVWGTF,
		ssaop.OpAMD64CMOVQGEF, ssaop.OpAMD64CMOVLGEF, ssaop.OpAMD64CMOVWGEF:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64CMOVQNEF, ssaop.OpAMD64CMOVLNEF, ssaop.OpAMD64CMOVWNEF:
		// Flag condition: ^ZERO || PARITY
		// Generate:
		//   CMOV*NE  SRC,DST
		//   CMOV*PS  SRC,DST
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		var q *obj.Prog
		if v.Op == ssaop.OpAMD64CMOVQNEF {
			q = s.Prog(x86.ACMOVQPS)
		} else if v.Op == ssaop.OpAMD64CMOVLNEF {
			q = s.Prog(x86.ACMOVLPS)
		} else {
			q = s.Prog(x86.ACMOVWPS)
		}
		q.From.Type = obj.TYPE_REG
		q.From.Reg = v.Args[1].Reg()
		q.To.Type = obj.TYPE_REG
		q.To.Reg = v.Reg()

	case ssaop.OpAMD64CMOVQEQF, ssaop.OpAMD64CMOVLEQF, ssaop.OpAMD64CMOVWEQF:
		// Flag condition: ZERO && !PARITY
		// Generate:
		//   MOV      SRC,TMP
		//   CMOV*NE  DST,TMP
		//   CMOV*PC  TMP,DST
		//
		// TODO(rasky): we could generate:
		//   CMOV*NE  DST,SRC
		//   CMOV*PC  SRC,DST
		// But this requires a way for regalloc to know that SRC might be
		// clobbered by this instruction.
		t := v.RegTmp()
		opregreg(s, moveByRegsWidth(t, v.Args[1].Reg(), v.Type.Size()), t, v.Args[1].Reg())

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = t
		var q *obj.Prog
		if v.Op == ssaop.OpAMD64CMOVQEQF {
			q = s.Prog(x86.ACMOVQPC)
		} else if v.Op == ssaop.OpAMD64CMOVLEQF {
			q = s.Prog(x86.ACMOVLPC)
		} else {
			q = s.Prog(x86.ACMOVWPC)
		}
		q.From.Type = obj.TYPE_REG
		q.From.Reg = t
		q.To.Type = obj.TYPE_REG
		q.To.Reg = v.Reg()

	case ssaop.OpAMD64MULQconst, ssaop.OpAMD64MULLconst:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.AddRestSourceReg(v.Args[0].Reg())

	case ssaop.OpAMD64ANDQconst:
		asm := v.Op.Asm()
		// If the constant is positive and fits into 32 bits, use ANDL.
		// This saves a few bytes of encoding.
		if 0 <= v.AuxInt && v.AuxInt <= (1<<32-1) {
			asm = x86.AANDL
		}
		p := s.Prog(asm)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64SUBQconst, ssaop.OpAMD64SUBLconst,
		ssaop.OpAMD64ANDLconst,
		ssaop.OpAMD64ORQconst, ssaop.OpAMD64ORLconst,
		ssaop.OpAMD64XORQconst, ssaop.OpAMD64XORLconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64SHLQconst, ssaop.OpAMD64SHLLconst,
		ssaop.OpAMD64SHRQconst, ssaop.OpAMD64SHRLconst, ssaop.OpAMD64SHRWconst, ssaop.OpAMD64SHRBconst,
		ssaop.OpAMD64SARQconst, ssaop.OpAMD64SARLconst, ssaop.OpAMD64SARWconst, ssaop.OpAMD64SARBconst,
		ssaop.OpAMD64ROLQconst, ssaop.OpAMD64ROLLconst, ssaop.OpAMD64ROLWconst, ssaop.OpAMD64ROLBconst:
		var maxShift int64
		switch v.Op {
		case ssaop.OpAMD64SHLQconst, ssaop.OpAMD64SHRQconst, ssaop.OpAMD64SARQconst, ssaop.OpAMD64ROLQconst:
			maxShift = 63
		case ssaop.OpAMD64SHLLconst, ssaop.OpAMD64SHRLconst, ssaop.OpAMD64SARLconst, ssaop.OpAMD64ROLLconst:
			maxShift = 31
		case ssaop.OpAMD64SHRWconst, ssaop.OpAMD64SARWconst, ssaop.OpAMD64ROLWconst:
			maxShift = 15
		case ssaop.OpAMD64SHRBconst, ssaop.OpAMD64SARBconst, ssaop.OpAMD64ROLBconst:
			maxShift = 7
		default:
			panic("unreachable")
		}
		if v.AuxInt < 0 || v.AuxInt > maxShift {
			v.Fatalf("shift amount out of range [0,%d]: %d", maxShift, v.AuxInt)
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64SBBQcarrymask, ssaop.OpAMD64SBBLcarrymask:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpAMD64LEAQ1, ssaop.OpAMD64LEAQ2, ssaop.OpAMD64LEAQ4, ssaop.OpAMD64LEAQ8,
		ssaop.OpAMD64LEAL1, ssaop.OpAMD64LEAL2, ssaop.OpAMD64LEAL4, ssaop.OpAMD64LEAL8,
		ssaop.OpAMD64LEAW1, ssaop.OpAMD64LEAW2, ssaop.OpAMD64LEAW4, ssaop.OpAMD64LEAW8:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64LEAQ, ssaop.OpAMD64LEAL, ssaop.OpAMD64LEAW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64CMPQ, ssaop.OpAMD64CMPL, ssaop.OpAMD64CMPW, ssaop.OpAMD64CMPB,
		ssaop.OpAMD64TESTQ, ssaop.OpAMD64TESTL, ssaop.OpAMD64TESTW, ssaop.OpAMD64TESTB,
		ssaop.OpAMD64BTL, ssaop.OpAMD64BTQ:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssaop.OpAMD64UCOMISS, ssaop.OpAMD64UCOMISD:
		// Go assembler has swapped operands for UCOMISx relative to CMP,
		// must account for that right here.
		opregreg(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg())
	case ssaop.OpAMD64CMPQconst, ssaop.OpAMD64CMPLconst, ssaop.OpAMD64CMPWconst, ssaop.OpAMD64CMPBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssaop.OpAMD64BTLconst, ssaop.OpAMD64BTQconst,
		ssaop.OpAMD64TESTQconst, ssaop.OpAMD64TESTLconst, ssaop.OpAMD64TESTWconst, ssaop.OpAMD64TESTBconst,
		ssaop.OpAMD64BTSQconst,
		ssaop.OpAMD64BTCQconst,
		ssaop.OpAMD64BTRQconst:
		op := v.Op
		if op == ssaop.OpAMD64BTQconst && v.AuxInt < 32 {
			// Emit 32-bit version because it's shorter
			op = ssaop.OpAMD64BTLconst
		}
		p := s.Prog(op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
	case ssaop.OpAMD64CMPQload, ssaop.OpAMD64CMPLload, ssaop.OpAMD64CMPWload, ssaop.OpAMD64CMPBload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[1].Reg()
	case ssaop.OpAMD64CMPQconstload, ssaop.OpAMD64CMPLconstload, ssaop.OpAMD64CMPWconstload, ssaop.OpAMD64CMPBconstload:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.From, v, sc.Off64())
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = sc.Val64()
	case ssaop.OpAMD64CMPQloadidx8, ssaop.OpAMD64CMPQloadidx1, ssaop.OpAMD64CMPLloadidx4, ssaop.OpAMD64CMPLloadidx1, ssaop.OpAMD64CMPWloadidx2, ssaop.OpAMD64CMPWloadidx1, ssaop.OpAMD64CMPBloadidx1:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[2].Reg()
	case ssaop.OpAMD64CMPQconstloadidx8, ssaop.OpAMD64CMPQconstloadidx1, ssaop.OpAMD64CMPLconstloadidx4, ssaop.OpAMD64CMPLconstloadidx1, ssaop.OpAMD64CMPWconstloadidx2, ssaop.OpAMD64CMPWconstloadidx1, ssaop.OpAMD64CMPBconstloadidx1:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux2(&p.From, v, sc.Off64())
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = sc.Val64()
	case ssaop.OpAMD64MOVLconst, ssaop.OpAMD64MOVQconst:
		x := v.Reg()

		// If flags aren't live (indicated by v.Aux == nil),
		// then we can rewrite MOV $0, AX into XOR AX, AX.
		if v.AuxInt == 0 && v.Aux == nil {
			opregreg(s, x86.AXORL, x, x)
			break
		}

		asm := v.Op.Asm()
		// Use MOVL to move a small constant into a register
		// when the constant is positive and fits into 32 bits.
		if 0 <= v.AuxInt && v.AuxInt <= (1<<32-1) {
			// The upper 32bit are zeroed automatically when using MOVL.
			asm = x86.AMOVL
		}
		p := s.Prog(asm)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x

	case ssaop.OpAMD64MOVSSconst, ssaop.OpAMD64MOVSDconst:
		x := v.Reg()
		if !isFPReg(x) && v.AuxInt == 0 && v.Aux == nil {
			opregreg(s, x86.AXORL, x, x)
			break
		}
		p := s.Prog(storeByRegWidth(x, v.Type.Size()))
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssaop.OpAMD64MOVQload, ssaop.OpAMD64MOVLload, ssaop.OpAMD64MOVWload, ssaop.OpAMD64MOVBload, ssaop.OpAMD64MOVOload,
		ssaop.OpAMD64MOVSSload, ssaop.OpAMD64MOVSDload, ssaop.OpAMD64MOVBQSXload, ssaop.OpAMD64MOVWQSXload, ssaop.OpAMD64MOVLQSXload,
		ssaop.OpAMD64MOVBEQload, ssaop.OpAMD64MOVBELload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64MOVBloadidx1, ssaop.OpAMD64MOVWloadidx1, ssaop.OpAMD64MOVLloadidx1, ssaop.OpAMD64MOVQloadidx1, ssaop.OpAMD64MOVSSloadidx1, ssaop.OpAMD64MOVSDloadidx1,
		ssaop.OpAMD64MOVQloadidx8, ssaop.OpAMD64MOVSDloadidx8, ssaop.OpAMD64MOVLloadidx8, ssaop.OpAMD64MOVLloadidx4, ssaop.OpAMD64MOVSSloadidx4, ssaop.OpAMD64MOVWloadidx2,
		ssaop.OpAMD64MOVBELloadidx1, ssaop.OpAMD64MOVBELloadidx4, ssaop.OpAMD64MOVBELloadidx8, ssaop.OpAMD64MOVBEQloadidx1, ssaop.OpAMD64MOVBEQloadidx8:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64MOVQstore, ssaop.OpAMD64MOVSSstore, ssaop.OpAMD64MOVSDstore, ssaop.OpAMD64MOVLstore, ssaop.OpAMD64MOVWstore, ssaop.OpAMD64MOVBstore, ssaop.OpAMD64MOVOstore,
		ssaop.OpAMD64ADDQmodify, ssaop.OpAMD64SUBQmodify, ssaop.OpAMD64ANDQmodify, ssaop.OpAMD64ORQmodify, ssaop.OpAMD64XORQmodify,
		ssaop.OpAMD64ADDLmodify, ssaop.OpAMD64SUBLmodify, ssaop.OpAMD64ANDLmodify, ssaop.OpAMD64ORLmodify, ssaop.OpAMD64XORLmodify,
		ssaop.OpAMD64MOVBEQstore, ssaop.OpAMD64MOVBELstore, ssaop.OpAMD64MOVBEWstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64MOVBstoreidx1, ssaop.OpAMD64MOVWstoreidx1, ssaop.OpAMD64MOVLstoreidx1, ssaop.OpAMD64MOVQstoreidx1, ssaop.OpAMD64MOVSSstoreidx1, ssaop.OpAMD64MOVSDstoreidx1,
		ssaop.OpAMD64MOVQstoreidx8, ssaop.OpAMD64MOVSDstoreidx8, ssaop.OpAMD64MOVLstoreidx8, ssaop.OpAMD64MOVSSstoreidx4, ssaop.OpAMD64MOVLstoreidx4, ssaop.OpAMD64MOVWstoreidx2,
		ssaop.OpAMD64ADDLmodifyidx1, ssaop.OpAMD64ADDLmodifyidx4, ssaop.OpAMD64ADDLmodifyidx8, ssaop.OpAMD64ADDQmodifyidx1, ssaop.OpAMD64ADDQmodifyidx8,
		ssaop.OpAMD64SUBLmodifyidx1, ssaop.OpAMD64SUBLmodifyidx4, ssaop.OpAMD64SUBLmodifyidx8, ssaop.OpAMD64SUBQmodifyidx1, ssaop.OpAMD64SUBQmodifyidx8,
		ssaop.OpAMD64ANDLmodifyidx1, ssaop.OpAMD64ANDLmodifyidx4, ssaop.OpAMD64ANDLmodifyidx8, ssaop.OpAMD64ANDQmodifyidx1, ssaop.OpAMD64ANDQmodifyidx8,
		ssaop.OpAMD64ORLmodifyidx1, ssaop.OpAMD64ORLmodifyidx4, ssaop.OpAMD64ORLmodifyidx8, ssaop.OpAMD64ORQmodifyidx1, ssaop.OpAMD64ORQmodifyidx8,
		ssaop.OpAMD64XORLmodifyidx1, ssaop.OpAMD64XORLmodifyidx4, ssaop.OpAMD64XORLmodifyidx8, ssaop.OpAMD64XORQmodifyidx1, ssaop.OpAMD64XORQmodifyidx8,
		ssaop.OpAMD64MOVBEWstoreidx1, ssaop.OpAMD64MOVBEWstoreidx2, ssaop.OpAMD64MOVBELstoreidx1, ssaop.OpAMD64MOVBELstoreidx4, ssaop.OpAMD64MOVBELstoreidx8, ssaop.OpAMD64MOVBEQstoreidx1, ssaop.OpAMD64MOVBEQstoreidx8:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		memIdx(&p.To, v)
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64ADDQconstmodify, ssaop.OpAMD64ADDLconstmodify,
		ssaop.OpAMD64ADDWconstmodify, ssaop.OpAMD64ADDBconstmodify:
		sc := v.AuxValAndOff()
		off := sc.Off64()
		val := sc.Val()
		if val == 1 || val == -1 {
			var asm obj.As
			switch v.Op {
			case ssaop.OpAMD64ADDQconstmodify:
				asm = x86.AINCQ
				if val == -1 {
					asm = x86.ADECQ
				}
			case ssaop.OpAMD64ADDLconstmodify:
				asm = x86.AINCL
				if val == -1 {
					asm = x86.ADECL
				}
			case ssaop.OpAMD64ADDWconstmodify:
				asm = x86.AINCW
				if val == -1 {
					asm = x86.ADECW
				}
			default:
				asm = x86.AINCB
				if val == -1 {
					asm = x86.ADECB
				}
			}
			p := s.Prog(asm)
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = v.Args[0].Reg()
			ssagen.AddAux2(&p.To, v, off)
			break
		}
		fallthrough
	case ssaop.OpAMD64ANDQconstmodify, ssaop.OpAMD64ANDLconstmodify, ssaop.OpAMD64ORQconstmodify, ssaop.OpAMD64ORLconstmodify,
		ssaop.OpAMD64XORQconstmodify, ssaop.OpAMD64XORLconstmodify,
		ssaop.OpAMD64ANDWconstmodify, ssaop.OpAMD64ANDBconstmodify, ssaop.OpAMD64ORWconstmodify, ssaop.OpAMD64ORBconstmodify,
		ssaop.OpAMD64XORWconstmodify, ssaop.OpAMD64XORBconstmodify,
		ssaop.OpAMD64BTSQconstmodify, ssaop.OpAMD64BTRQconstmodify, ssaop.OpAMD64BTCQconstmodify:
		sc := v.AuxValAndOff()
		off := sc.Off64()
		val := sc.Val64()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = val
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, off)

	case ssaop.OpAMD64MOVQstoreconst, ssaop.OpAMD64MOVLstoreconst, ssaop.OpAMD64MOVWstoreconst, ssaop.OpAMD64MOVBstoreconst:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		if sc.Val() == 0 && s.ABI == obj.ABIInternal && buildcfg.GOOS != "plan9" && (v.Op == ssaop.OpAMD64MOVQstoreconst || v.Op == ssaop.OpAMD64MOVLstoreconst) {
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x86.REG_X15
		} else {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = sc.Val64()
		}
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssaop.OpAMD64MOVOstoreconst:
		sc := v.AuxValAndOff()
		if sc.Val() != 0 {
			v.Fatalf("MOVO for non zero constants not implemented: %s", v.LongString())
		}

		if s.ABI != obj.ABIInternal {
			// zero X15 manually
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_X15
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())

	case ssaop.OpAMD64MOVQstoreconstidx1, ssaop.OpAMD64MOVQstoreconstidx8, ssaop.OpAMD64MOVLstoreconstidx1, ssaop.OpAMD64MOVLstoreconstidx4, ssaop.OpAMD64MOVWstoreconstidx1, ssaop.OpAMD64MOVWstoreconstidx2, ssaop.OpAMD64MOVBstoreconstidx1,
		ssaop.OpAMD64ADDLconstmodifyidx1, ssaop.OpAMD64ADDLconstmodifyidx4, ssaop.OpAMD64ADDLconstmodifyidx8, ssaop.OpAMD64ADDQconstmodifyidx1, ssaop.OpAMD64ADDQconstmodifyidx8,
		ssaop.OpAMD64ANDLconstmodifyidx1, ssaop.OpAMD64ANDLconstmodifyidx4, ssaop.OpAMD64ANDLconstmodifyidx8, ssaop.OpAMD64ANDQconstmodifyidx1, ssaop.OpAMD64ANDQconstmodifyidx8,
		ssaop.OpAMD64ORLconstmodifyidx1, ssaop.OpAMD64ORLconstmodifyidx4, ssaop.OpAMD64ORLconstmodifyidx8, ssaop.OpAMD64ORQconstmodifyidx1, ssaop.OpAMD64ORQconstmodifyidx8,
		ssaop.OpAMD64XORLconstmodifyidx1, ssaop.OpAMD64XORLconstmodifyidx4, ssaop.OpAMD64XORLconstmodifyidx8, ssaop.OpAMD64XORQconstmodifyidx1, ssaop.OpAMD64XORQconstmodifyidx8,
		ssaop.OpAMD64ADDWconstmodifyidx1, ssaop.OpAMD64ADDWconstmodifyidx2, ssaop.OpAMD64ADDBconstmodifyidx1,
		ssaop.OpAMD64ANDWconstmodifyidx1, ssaop.OpAMD64ANDWconstmodifyidx2, ssaop.OpAMD64ANDBconstmodifyidx1,
		ssaop.OpAMD64ORWconstmodifyidx1, ssaop.OpAMD64ORWconstmodifyidx2, ssaop.OpAMD64ORBconstmodifyidx1,
		ssaop.OpAMD64XORWconstmodifyidx1, ssaop.OpAMD64XORWconstmodifyidx2, ssaop.OpAMD64XORBconstmodifyidx1:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		if sc.Val() == 0 && s.ABI == obj.ABIInternal && buildcfg.GOOS != "plan9" {
			switch v.Op {
			case ssaop.OpAMD64MOVQstoreconstidx1, ssaop.OpAMD64MOVQstoreconstidx8,
				ssaop.OpAMD64MOVLstoreconstidx1, ssaop.OpAMD64MOVLstoreconstidx4:
				p.From.Type = obj.TYPE_REG
				p.From.Reg = x86.REG_X15
			}
		}
		switch {
		case p.As == x86.AADDQ && p.From.Offset == 1:
			p.As = x86.AINCQ
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDQ && p.From.Offset == -1:
			p.As = x86.ADECQ
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDL && p.From.Offset == 1:
			p.As = x86.AINCL
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDL && p.From.Offset == -1:
			p.As = x86.ADECL
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDW && p.From.Offset == 1:
			p.As = x86.AINCW
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDW && p.From.Offset == -1:
			p.As = x86.ADECW
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDB && p.From.Offset == 1:
			p.As = x86.AINCB
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDB && p.From.Offset == -1:
			p.As = x86.ADECB
			p.From.Type = obj.TYPE_NONE
		}
		memIdx(&p.To, v)
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssaop.OpAMD64MOVLQSX, ssaop.OpAMD64MOVWQSX, ssaop.OpAMD64MOVBQSX, ssaop.OpAMD64MOVLQZX, ssaop.OpAMD64MOVWQZX, ssaop.OpAMD64MOVBQZX,
		ssaop.OpAMD64CVTTSS2SL, ssaop.OpAMD64CVTTSD2SL, ssaop.OpAMD64CVTTSS2SQ, ssaop.OpAMD64CVTTSD2SQ,
		ssaop.OpAMD64CVTSS2SD, ssaop.OpAMD64CVTSD2SS, ssaop.OpAMD64VPBROADCASTB, ssaop.OpAMD64PMOVMSKB:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssaop.OpAMD64CVTSL2SD, ssaop.OpAMD64CVTSQ2SD, ssaop.OpAMD64CVTSQ2SS, ssaop.OpAMD64CVTSL2SS:
		r := v.Reg()
		// Break false dependency on destination register.
		opregreg(s, x86.AXORPS, r, r)
		opregreg(s, v.Op.Asm(), r, v.Args[0].Reg())
	case ssaop.OpAMD64MOVQi2f, ssaop.OpAMD64MOVQf2i, ssaop.OpAMD64MOVLi2f, ssaop.OpAMD64MOVLf2i:
		var p *obj.Prog
		switch v.Op {
		case ssaop.OpAMD64MOVQi2f, ssaop.OpAMD64MOVQf2i:
			p = s.Prog(x86.AMOVQ)
		case ssaop.OpAMD64MOVLi2f, ssaop.OpAMD64MOVLf2i:
			p = s.Prog(x86.AMOVL)
		}
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64ADDQload, ssaop.OpAMD64ADDLload, ssaop.OpAMD64SUBQload, ssaop.OpAMD64SUBLload,
		ssaop.OpAMD64ANDQload, ssaop.OpAMD64ANDLload, ssaop.OpAMD64ORQload, ssaop.OpAMD64ORLload,
		ssaop.OpAMD64XORQload, ssaop.OpAMD64XORLload, ssaop.OpAMD64ADDSDload, ssaop.OpAMD64ADDSSload,
		ssaop.OpAMD64SUBSDload, ssaop.OpAMD64SUBSSload, ssaop.OpAMD64MULSDload, ssaop.OpAMD64MULSSload,
		ssaop.OpAMD64DIVSDload, ssaop.OpAMD64DIVSSload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64ADDLloadidx1, ssaop.OpAMD64ADDLloadidx4, ssaop.OpAMD64ADDLloadidx8, ssaop.OpAMD64ADDQloadidx1, ssaop.OpAMD64ADDQloadidx8,
		ssaop.OpAMD64SUBLloadidx1, ssaop.OpAMD64SUBLloadidx4, ssaop.OpAMD64SUBLloadidx8, ssaop.OpAMD64SUBQloadidx1, ssaop.OpAMD64SUBQloadidx8,
		ssaop.OpAMD64ANDLloadidx1, ssaop.OpAMD64ANDLloadidx4, ssaop.OpAMD64ANDLloadidx8, ssaop.OpAMD64ANDQloadidx1, ssaop.OpAMD64ANDQloadidx8,
		ssaop.OpAMD64ORLloadidx1, ssaop.OpAMD64ORLloadidx4, ssaop.OpAMD64ORLloadidx8, ssaop.OpAMD64ORQloadidx1, ssaop.OpAMD64ORQloadidx8,
		ssaop.OpAMD64XORLloadidx1, ssaop.OpAMD64XORLloadidx4, ssaop.OpAMD64XORLloadidx8, ssaop.OpAMD64XORQloadidx1, ssaop.OpAMD64XORQloadidx8,
		ssaop.OpAMD64ADDSSloadidx1, ssaop.OpAMD64ADDSSloadidx4, ssaop.OpAMD64ADDSDloadidx1, ssaop.OpAMD64ADDSDloadidx8,
		ssaop.OpAMD64SUBSSloadidx1, ssaop.OpAMD64SUBSSloadidx4, ssaop.OpAMD64SUBSDloadidx1, ssaop.OpAMD64SUBSDloadidx8,
		ssaop.OpAMD64MULSSloadidx1, ssaop.OpAMD64MULSSloadidx4, ssaop.OpAMD64MULSDloadidx1, ssaop.OpAMD64MULSDloadidx8,
		ssaop.OpAMD64DIVSSloadidx1, ssaop.OpAMD64DIVSSloadidx4, ssaop.OpAMD64DIVSDloadidx1, ssaop.OpAMD64DIVSDloadidx8:
		p := s.Prog(v.Op.Asm())

		r, i := v.Args[1].Reg(), v.Args[2].Reg()
		p.From.Type = obj.TYPE_MEM
		p.From.Scale = v.Op.Scale()
		if p.From.Scale == 1 && i == x86.REG_SP {
			r, i = i, r
		}
		p.From.Reg = r
		p.From.Index = i

		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64LoweredZero:
		if s.ABI != obj.ABIInternal {
			// zero X15 manually
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
		}
		ptrReg := v.Args[0].Reg()
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Zero too small %d", n)
		}
		zero16 := func(off int64) {
			zero16(s, ptrReg, off)
		}

		// Generate zeroing instructions.
		var off int64
		for n >= 16 {
			zero16(off)
			off += 16
			n -= 16
		}
		if n != 0 {
			// use partially overlapped write.
			// TODO: n <= 8, use smaller write?
			zero16(off + n - 16)
		}

	case ssaop.OpAMD64LoweredZeroLoop:
		if s.ABI != obj.ABIInternal {
			// zero X15 manually
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
		}
		ptrReg := v.Args[0].Reg()
		countReg := v.RegTmp()
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     4 instructions to implement the loop
			//     4 instructions in the loop body
			//   vs
			//     8 instructions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("ZeroLoop size too small %d", n)
		}
		zero16 := func(off int64) {
			zero16(s, ptrReg, off)
		}

		// Put iteration count in a register.
		//   MOVL    $n, countReg
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Zero loopSize bytes starting at ptrReg.
		for i := range loopSize / 16 {
			zero16(i * 16)
		}
		//   ADDQ    $loopSize, ptrReg
		p = s.Prog(x86.AADDQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ptrReg
		//   DECL    countReg
		p = s.Prog(x86.ADECL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		// Jump to first instruction in loop if we're not done yet.
		//   JNE     head
		p = s.Prog(x86.AJNE)
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		// Write any fractional portion.
		var off int64
		for n >= 16 {
			zero16(off)
			off += 16
			n -= 16
		}
		if n != 0 {
			// Use partially-overlapping write.
			// TODO: n <= 8, use smaller write?
			zero16(off + n - 16)
		}

	case ssaop.OpAMD64LoweredMove:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		tmpReg := int16(x86.REG_X14)
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Move too small %d", n)
		}
		// move 16 bytes from srcReg+off to dstReg+off.
		move16 := func(off int64) {
			move16(s, srcReg, dstReg, tmpReg, off)
		}

		// Generate copying instructions.
		var off int64
		for n >= 16 {
			move16(off)
			off += 16
			n -= 16
		}
		if n != 0 {
			// use partially overlapped read/write.
			// TODO: use smaller operations when we can?
			move16(off + n - 16)
		}

	case ssaop.OpAMD64LoweredMoveLoop:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		countReg := v.RegTmp()
		tmpReg := int16(x86.REG_X14)
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     4 instructions to implement the loop
			//     4 instructions in the loop body
			//   vs
			//     8 instructions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("ZeroLoop size too small %d", n)
		}
		// move 16 bytes from srcReg+off to dstReg+off.
		move16 := func(off int64) {
			move16(s, srcReg, dstReg, tmpReg, off)
		}

		// Put iteration count in a register.
		//   MOVL    $n, countReg
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Copy loopSize bytes starting at srcReg to dstReg.
		for i := range loopSize / 16 {
			move16(i * 16)
		}
		//   ADDQ    $loopSize, srcReg
		p = s.Prog(x86.AADDQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = srcReg
		//   ADDQ    $loopSize, dstReg
		p = s.Prog(x86.AADDQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = dstReg
		//   DECL    countReg
		p = s.Prog(x86.ADECL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		// Jump to loop header if we're not done yet.
		//   JNE     head
		p = s.Prog(x86.AJNE)
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		// Copy any fractional portion.
		var off int64
		for n >= 16 {
			move16(off)
			off += 16
			n -= 16
		}
		if n != 0 {
			// Use partially-overlapping copy.
			move16(off + n - 16)
		}

	case ssaop.OpCopy: // TODO: use MOVQreg for reg->reg copies instead of OpCopy?
		if v.Type.IsMemory() {
			return
		}
		arg := v.Args[0]
		x := arg.Reg()
		y := v.Reg()
		if v.Type.IsSIMD() {
			x = simdOrMaskReg(arg)
			y = simdOrMaskReg(v)
		}
		if x != y {
			width := v.Type.Size()
			if width == 8 && isGPReg(y) && ssacore.ZeroUpper32Bits(arg) {
				// The source was naturally zext-ed from 32 to 64 bits,
				// but we are asked to do a full 64-bit copy.
				// Save the REX prefix byte in I-CACHE by using a 32-bit move,
				// since it zeroes the upper 32 bits anyway.
				width = 4
			}
			opregreg(s, moveByRegsWidth(y, x, width), y, x)
		}
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		r := v.Reg()
		p := s.Prog(loadByRegWidth(r, v.Type.Size()))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		if v.Type.IsSIMD() {
			r = simdOrMaskReg(v)
		}
		p.To.Reg = r

	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		if v.Type.IsSIMD() {
			r = simdOrMaskReg(v.Args[0])
		}
		p := s.Prog(storeByRegWidth(r, v.Type.Size()))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		ssagen.AddrAuto(&p.To, v)
	case ssaop.OpAMD64LoweredHasCPUFeature:
		// If this load changes width, update zeroUpperBits in AMD64Ops.go.
		p := s.Prog(x86.AMOVBLZX)
		p.From.Type = obj.TYPE_MEM
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		for _, ap := range v.Block.Func.RegArgs {
			// Pass the spill/unspill information along to the assembler, offset by size of return PC pushed on stack.
			addr := ssagen.SpillSlotAddr(ap, x86.REG_SP, v.Block.Func.Config.PtrSize)
			reg := ap.Reg
			t := ap.Type
			sz := t.Size()
			if t.IsSIMD() {
				reg = simdRegBySize(reg, sz)
			}
			s.FuncInfo().AddSpill(
				obj.RegSpill{Reg: reg, Addr: addr, Unspill: loadByRegWidth(reg, sz), Spill: storeByRegWidth(reg, sz)})
		}
		v.Block.Func.RegArgs = nil
		ssagen.CheckArgReg(v)
	case ssaop.OpAMD64LoweredGetClosurePtr:
		// Closure pointer is DX.
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpAMD64LoweredGetG:
		if s.ABI == obj.ABIInternal {
			v.Fatalf("LoweredGetG should not appear in ABIInternal")
		}
		r := v.Reg()
		getgFromTLS(s, r)
	case ssaop.OpAMD64CALLstatic, ssaop.OpAMD64CALLtail, ssaop.OpAMD64CALLtailinter:
		if s.ABI == obj.ABI0 && v.Aux.(*ssacore.AuxCall).Fn.ABI() == obj.ABIInternal {
			// zeroing X15 when entering ABIInternal from ABI0
			zeroX15(s)
			// set G register from TLS
			getgFromTLS(s, x86.REG_R14)
		}
		if v.Op == ssaop.OpAMD64CALLtail || v.Op == ssaop.OpAMD64CALLtailinter {
			s.TailCall(v)
			break
		}
		s.Call(v)
		if s.ABI == obj.ABIInternal && v.Aux.(*ssacore.AuxCall).Fn.ABI() == obj.ABI0 {
			// zeroing X15 when entering ABIInternal from ABI0
			zeroX15(s)
			// set G register from TLS
			getgFromTLS(s, x86.REG_R14)
		}
	case ssaop.OpAMD64CALLclosure, ssaop.OpAMD64CALLinter:
		s.Call(v)

	case ssaop.OpAMD64LoweredGetCallerPC:
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = -8 // PC is stored 8 bytes below first parameter.
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64LoweredGetCallerSP:
		// caller's SP is the address of the first arg
		mov := x86.AMOVQ
		if types.PtrSize == 4 {
			mov = x86.AMOVL
		}
		p := s.Prog(mov)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize // 0 on amd64, just to be consistent with other architectures
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpAMD64LoweredPanicBoundsRR, ssaop.OpAMD64LoweredPanicBoundsRC, ssaop.OpAMD64LoweredPanicBoundsCR, ssaop.OpAMD64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssacore.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpAMD64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - x86.REG_AX)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - x86.REG_AX)
		case ssaop.OpAMD64LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - x86.REG_AX)
			c := v.Aux.(ssacore.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(x86.AMOVQ)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(yVal)
			}
		case ssaop.OpAMD64LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - x86.REG_AX)
			c := v.Aux.(ssacore.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(x86.AMOVQ)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(xVal)
			}
		case ssaop.OpAMD64LoweredPanicBoundsCC:
			c := v.Aux.(ssacore.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(x86.AMOVQ)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(xVal)
			}
			c = v.Aux.(ssacore.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(x86.AMOVQ)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(yVal)
			}
		}
		c := abi.BoundsEncode(code, signed, xIsReg, yIsReg, xVal, yVal)

		p := s.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_PanicBounds)
		p.To.SetConst(int64(c))
		p = s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.PanicBounds

	case ssaop.OpAMD64NEGQ, ssaop.OpAMD64NEGL,
		ssaop.OpAMD64BSWAPQ, ssaop.OpAMD64BSWAPL,
		ssaop.OpAMD64NOTQ, ssaop.OpAMD64NOTL:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64NEGLflags:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpAMD64ADDQconstflags, ssaop.OpAMD64ADDLconstflags:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		// Note: the inc/dec instructions do not modify
		// the carry flag like add$1 / sub$1 do.
		// We currently never use the CF/OF flags from
		// these instructions, so that is ok.
		switch {
		case p.As == x86.AADDQ && p.From.Offset == 1:
			p.As = x86.AINCQ
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDQ && p.From.Offset == -1:
			p.As = x86.ADECQ
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDL && p.From.Offset == 1:
			p.As = x86.AINCL
			p.From.Type = obj.TYPE_NONE
		case p.As == x86.AADDL && p.From.Offset == -1:
			p.As = x86.ADECL
			p.From.Type = obj.TYPE_NONE
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpAMD64BSFQ, ssaop.OpAMD64BSRQ, ssaop.OpAMD64BSFL, ssaop.OpAMD64BSRL, ssaop.OpAMD64SQRTSD, ssaop.OpAMD64SQRTSS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		switch v.Op {
		case ssaop.OpAMD64BSFQ, ssaop.OpAMD64BSRQ:
			p.To.Reg = v.Reg0()
		case ssaop.OpAMD64BSFL, ssaop.OpAMD64BSRL, ssaop.OpAMD64SQRTSD, ssaop.OpAMD64SQRTSS:
			p.To.Reg = v.Reg()
		}
	case ssaop.OpAMD64LoweredRound32F, ssaop.OpAMD64LoweredRound64F:
		// input is already rounded
	case ssaop.OpAMD64ROUNDSD, ssaop.OpAMD64ROUNDSS:
		p := s.Prog(v.Op.Asm())
		val := v.AuxInt
		// 0 means math.RoundToEven, 1 Floor, 2 Ceil, 3 Trunc
		if val < 0 || val > 3 {
			v.Fatalf("Invalid rounding mode")
		}
		p.From.Offset = val
		p.From.Type = obj.TYPE_CONST
		p.AddRestSourceReg(v.Args[0].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64POPCNTQ, ssaop.OpAMD64POPCNTL,
		ssaop.OpAMD64TZCNTQ, ssaop.OpAMD64TZCNTL,
		ssaop.OpAMD64LZCNTQ, ssaop.OpAMD64LZCNTL:
		if v.Args[0].Reg() != v.Reg() {
			// POPCNT/TZCNT/LZCNT have a false dependency on the destination register on Intel cpus.
			// TZCNT/LZCNT problem affects pre-Skylake models. See discussion at https://gcc.gnu.org/bugzilla/show_bug.cgi?id=62011#c7.
			// Xor register with itself to break the dependency.
			opregreg(s, x86.AXORL, v.Reg(), v.Reg())
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64SETEQ, ssaop.OpAMD64SETNE,
		ssaop.OpAMD64SETL, ssaop.OpAMD64SETLE,
		ssaop.OpAMD64SETG, ssaop.OpAMD64SETGE,
		ssaop.OpAMD64SETGF, ssaop.OpAMD64SETGEF,
		ssaop.OpAMD64SETB, ssaop.OpAMD64SETBE,
		ssaop.OpAMD64SETORD, ssaop.OpAMD64SETNAN,
		ssaop.OpAMD64SETA, ssaop.OpAMD64SETAE,
		ssaop.OpAMD64SETO:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64SETEQstore, ssaop.OpAMD64SETNEstore,
		ssaop.OpAMD64SETLstore, ssaop.OpAMD64SETLEstore,
		ssaop.OpAMD64SETGstore, ssaop.OpAMD64SETGEstore,
		ssaop.OpAMD64SETBstore, ssaop.OpAMD64SETBEstore,
		ssaop.OpAMD64SETAstore, ssaop.OpAMD64SETAEstore:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)

	case ssaop.OpAMD64SETEQstoreidx1, ssaop.OpAMD64SETNEstoreidx1,
		ssaop.OpAMD64SETLstoreidx1, ssaop.OpAMD64SETLEstoreidx1,
		ssaop.OpAMD64SETGstoreidx1, ssaop.OpAMD64SETGEstoreidx1,
		ssaop.OpAMD64SETBstoreidx1, ssaop.OpAMD64SETBEstoreidx1,
		ssaop.OpAMD64SETAstoreidx1, ssaop.OpAMD64SETAEstoreidx1:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.To, v)
		ssagen.AddAux(&p.To, v)

	case ssaop.OpAMD64SETNEF:
		t := v.RegTmp()
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPS)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = t
		// ORL avoids partial register write and is smaller than ORQ, used by old compiler
		opregreg(s, x86.AORL, v.Reg(), t)

	case ssaop.OpAMD64SETEQF:
		t := v.RegTmp()
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPC)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = t
		// ANDL avoids partial register write and is smaller than ANDQ, used by old compiler
		opregreg(s, x86.AANDL, v.Reg(), t)

	case ssaop.OpAMD64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssaop.OpAMD64FlagEQ, ssaop.OpAMD64FlagLT_ULT, ssaop.OpAMD64FlagLT_UGT, ssaop.OpAMD64FlagGT_ULT, ssaop.OpAMD64FlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssaop.OpAMD64AddTupleFirst32, ssaop.OpAMD64AddTupleFirst64:
		v.Fatalf("AddTupleFirst* should never make it to codegen %v", v.LongString())
	case ssaop.OpAMD64REPSTOSQ:
		s.Prog(x86.AREP)
		s.Prog(x86.ASTOSQ)
	case ssaop.OpAMD64REPMOVSQ:
		s.Prog(x86.AREP)
		s.Prog(x86.AMOVSQ)
	case ssaop.OpAMD64LoweredNilCheck:
		// Issue a load which will fault if the input is nil.
		// TODO: We currently use the 2-byte instruction TESTB AX, (reg).
		// Should we use the 3-byte TESTB $0, (reg) instead? It is larger
		// but it doesn't have false dependency on AX.
		// Or maybe allocate an output register and use MOVL (reg),reg2 ?
		// That trades clobbering flags for clobbering a register.
		p := s.Prog(x86.ATESTB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_AX
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssaop.OpAMD64MOVBatomicload, ssaop.OpAMD64MOVLatomicload, ssaop.OpAMD64MOVQatomicload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpAMD64XCHGB, ssaop.OpAMD64XCHGL, ssaop.OpAMD64XCHGQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg0()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64XADDLlock, ssaop.OpAMD64XADDQlock:
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg0()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64CMPXCHGLlock, ssaop.OpAMD64CMPXCHGQlock:
		if v.Args[1].Reg() != x86.REG_AX {
			v.Fatalf("input[1] not in AX %s", v.LongString())
		}
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
		p = s.Prog(x86.ASETEQ)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpAMD64ANDBlock, ssaop.OpAMD64ANDLlock, ssaop.OpAMD64ANDQlock,
		ssaop.OpAMD64ORBlock, ssaop.OpAMD64ORLlock, ssaop.OpAMD64ORQlock,
		ssaop.OpAMD64ADDLlock, ssaop.OpAMD64ADDQlock,
		ssaop.OpAMD64SUBLlock, ssaop.OpAMD64SUBQlock:
		// Atomic memory operations that don't need to return the old value.
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64INCLlock, ssaop.OpAMD64INCQlock,
		ssaop.OpAMD64DECLlock, ssaop.OpAMD64DECQlock:
		// Unary atomic memory operations that don't need to return the old value.
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpAMD64LoweredAtomicAnd64, ssaop.OpAMD64LoweredAtomicOr64, ssaop.OpAMD64LoweredAtomicAnd32, ssaop.OpAMD64LoweredAtomicOr32:
		// Atomic memory operations that need to return the old value.
		// We need to do these with compare-and-exchange to get access to the old value.
		// loop:
		// MOVQ mask, tmp
		// MOVQ (addr), AX
		// ANDQ AX, tmp
		// LOCK CMPXCHGQ tmp, (addr) : note that AX is implicit old value to compare against
		// JNE loop
		// : result in AX
		//
		// If the width written to AX changes, update zeroUpperBits in AMD64Ops.go.
		mov := x86.AMOVQ
		op := x86.AANDQ
		cmpxchg := x86.ACMPXCHGQ
		switch v.Op {
		case ssaop.OpAMD64LoweredAtomicOr64:
			op = x86.AORQ
		case ssaop.OpAMD64LoweredAtomicAnd32:
			mov = x86.AMOVL
			op = x86.AANDL
			cmpxchg = x86.ACMPXCHGL
		case ssaop.OpAMD64LoweredAtomicOr32:
			mov = x86.AMOVL
			op = x86.AORL
			cmpxchg = x86.ACMPXCHGL
		}
		addr := v.Args[0].Reg()
		mask := v.Args[1].Reg()
		tmp := v.RegTmp()
		p1 := s.Prog(mov)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = mask
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = tmp
		p2 := s.Prog(mov)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Reg = addr
		ssagen.AddAux(&p2.From, v)
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = x86.REG_AX
		p3 := s.Prog(op)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = x86.REG_AX
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = tmp
		s.Prog(x86.ALOCK)
		p5 := s.Prog(cmpxchg)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = tmp
		p5.To.Type = obj.TYPE_MEM
		p5.To.Reg = addr
		ssagen.AddAux(&p5.To, v)
		p6 := s.Prog(x86.AJNE)
		p6.To.Type = obj.TYPE_BRANCH
		p6.To.SetTarget(p1)
	case ssaop.OpAMD64PrefetchT0, ssaop.OpAMD64PrefetchNTA:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
	case ssaop.OpClobber:
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0xdeaddead
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = x86.REG_SP
		ssagen.AddAux(&p.To, v)
		p = s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0xdeaddead
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = x86.REG_SP
		ssagen.AddAux(&p.To, v)
		p.To.Offset += 4
	case ssaop.OpClobberReg:
		x := uint64(0xdeaddeaddeaddead)
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(x)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	// SIMD ops
	case ssaop.OpAMD64VZEROUPPER, ssaop.OpAMD64VZEROALL:
		s.Prog(v.Op.Asm())

	case ssaop.OpAMD64Zero128: // no code emitted

	case ssaop.OpAMD64Zero256, ssaop.OpAMD64Zero512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v)
		p.AddRestSourceReg(simdReg(v))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssaop.OpAMD64VMOVSSf2v, ssaop.OpAMD64VMOVSDf2v:
		// These are for initializing the least 32/64 bits of a SIMD register from a "float".
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.AddRestSourceReg(x86.REG_X15)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssaop.OpAMD64VMOVQload, ssaop.OpAMD64VMOVDload,
		ssaop.OpAMD64VMOVSSload, ssaop.OpAMD64VMOVSDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssaop.OpAMD64VMOVSSconst, ssaop.OpAMD64VMOVSDconst:
		// for loading constants directly into SIMD registers
		x := simdReg(v)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x

	case ssaop.OpAMD64VMOVD, ssaop.OpAMD64VMOVQ:
		// These are for initializing the least 32/64 bits of a SIMD register from an "int".
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssaop.OpAMD64VMOVDQUload128, ssaop.OpAMD64VMOVDQUload256, ssaop.OpAMD64VMOVDQUload512,
		ssaop.OpAMD64KMOVBload, ssaop.OpAMD64KMOVWload, ssaop.OpAMD64KMOVDload, ssaop.OpAMD64KMOVQload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdOrMaskReg(v)
	case ssaop.OpAMD64VMOVDQUstore128, ssaop.OpAMD64VMOVDQUstore256, ssaop.OpAMD64VMOVDQUstore512,
		ssaop.OpAMD64KMOVBstore, ssaop.OpAMD64KMOVWstore, ssaop.OpAMD64KMOVDstore, ssaop.OpAMD64KMOVQstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdOrMaskReg(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)

	case ssaop.OpAMD64VPMASK32load128, ssaop.OpAMD64VPMASK64load128, ssaop.OpAMD64VPMASK32load256, ssaop.OpAMD64VPMASK64load256:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)
		p.AddRestSourceReg(simdReg(v.Args[1])) // masking simd reg

	case ssaop.OpAMD64VPMASK32store128, ssaop.OpAMD64VPMASK64store128, ssaop.OpAMD64VPMASK32store256, ssaop.OpAMD64VPMASK64store256:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
		p.AddRestSourceReg(simdReg(v.Args[1])) // masking simd reg

	case ssaop.OpAMD64VPMASK64load512, ssaop.OpAMD64VPMASK32load512, ssaop.OpAMD64VPMASK16load512, ssaop.OpAMD64VPMASK8load512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)
		p.AddRestSourceReg(v.Args[1].Reg()) // simd mask reg
		x86.ParseSuffix(p, "Z")             // must be zero if not in mask

	case ssaop.OpAMD64KANDB, ssaop.OpAMD64KANDW, ssaop.OpAMD64KANDD, ssaop.OpAMD64KANDQ,
		ssaop.OpAMD64KORB, ssaop.OpAMD64KORW, ssaop.OpAMD64KORD, ssaop.OpAMD64KORQ,
		ssaop.OpAMD64KXORB, ssaop.OpAMD64KXORW, ssaop.OpAMD64KXORD, ssaop.OpAMD64KXORQ,
		ssaop.OpAMD64KXNORB, ssaop.OpAMD64KXNORW, ssaop.OpAMD64KXNORD, ssaop.OpAMD64KXNORQ: // XNOR == EQ
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.AddRestSourceReg(v.Args[1].Reg()) // masking simd reg

	case ssaop.OpAMD64VPMASK64store512, ssaop.OpAMD64VPMASK32store512, ssaop.OpAMD64VPMASK16store512, ssaop.OpAMD64VPMASK8store512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
		p.AddRestSourceReg(v.Args[1].Reg()) // simd mask reg

	case ssaop.OpAMD64VPMOVMToVec8x16,
		ssaop.OpAMD64VPMOVMToVec8x32,
		ssaop.OpAMD64VPMOVMToVec8x64,
		ssaop.OpAMD64VPMOVMToVec16x8,
		ssaop.OpAMD64VPMOVMToVec16x16,
		ssaop.OpAMD64VPMOVMToVec16x32,
		ssaop.OpAMD64VPMOVMToVec32x4,
		ssaop.OpAMD64VPMOVMToVec32x8,
		ssaop.OpAMD64VPMOVMToVec32x16,
		ssaop.OpAMD64VPMOVMToVec64x2,
		ssaop.OpAMD64VPMOVMToVec64x4,
		ssaop.OpAMD64VPMOVMToVec64x8:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssaop.OpAMD64VPMOVVec8x16ToM,
		ssaop.OpAMD64VPMOVVec8x32ToM,
		ssaop.OpAMD64VPMOVVec8x64ToM,
		ssaop.OpAMD64VPMOVVec16x8ToM,
		ssaop.OpAMD64VPMOVVec16x16ToM,
		ssaop.OpAMD64VPMOVVec16x32ToM,
		ssaop.OpAMD64VPMOVVec32x4ToM,
		ssaop.OpAMD64VPMOVVec32x8ToM,
		ssaop.OpAMD64VPMOVVec32x16ToM,
		ssaop.OpAMD64VPMOVVec64x2ToM,
		ssaop.OpAMD64VPMOVVec64x4ToM,
		ssaop.OpAMD64VPMOVVec64x8ToM,
		ssaop.OpAMD64VPMOVMSKB128,
		ssaop.OpAMD64VPMOVMSKB256,
		ssaop.OpAMD64VMOVMSKPS128,
		ssaop.OpAMD64VMOVMSKPS256,
		ssaop.OpAMD64VMOVMSKPD128,
		ssaop.OpAMD64VMOVMSKPD256:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpAMD64KMOVQk, ssaop.OpAMD64KMOVDk, ssaop.OpAMD64KMOVWk, ssaop.OpAMD64KMOVBk,
		ssaop.OpAMD64KMOVQi, ssaop.OpAMD64KMOVDi, ssaop.OpAMD64KMOVWi, ssaop.OpAMD64KMOVBi:
		// See also ssa.OpAMD64KMOVQload
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpAMD64VPTEST:
		// Some instructions setting flags put their second operand into the destination reg.
		// See also CMP[BWDQ].
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v.Args[1])

	default:
		if !ssaGenSIMDValue(s, v) {
			v.Fatalf("genValue not implemented: %s", v.LongString())
		}
	}
}

// zeroX15 zeroes the X15 register.
func zeroX15(s *ssagen.State) {
	opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
}

// Example instruction: VRSQRTPS X1, X1
func simdV11(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[0])
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSUBD X1, X2, X3
func simdV21(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	// Vector registers operands follows a right-to-left order.
	// e.g. VPSUBD X1, X2, X3 means X3 = X2 - X1.
	p.From.Reg = simdReg(v.Args[1])
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// This function is to accustomize the shifts.
// The 2nd arg is an XMM, and this function merely checks that.
// Example instruction: VPSLLQ Z1, X1, Z2
func simdVfpv(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	// Vector registers operands follows a right-to-left order.
	// e.g. VPSUBD X1, X2, X3 means X3 = X2 - X1.
	p.From.Reg = v.Args[1].Reg()
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPCMPEQW Z26, Z30, K4
func simdV2k(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VPMINUQ X21, X3, K3, X31
func simdV2kv(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	p.AddRestSourceReg(simdReg(v.Args[0]))
	// These "simd*" series of functions assumes:
	// Any "K" register that serves as the write-mask
	// or "predicate" for "predicated AVX512 instructions"
	// sits right at the end of the operand list.
	// TODO: verify this assumption.
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPABSB X1, X2, K3 (masking merging)
func simdV2kvResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	// These "simd*" series of functions assumes:
	// Any "K" register that serves as the write-mask
	// or "predicate" for "predicated AVX512 instructions"
	// sits right at the end of the operand list.
	// TODO: verify this assumption.
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// This function is to accustomize the shifts.
// The 2nd arg is an XMM, and this function merely checks that.
// Example instruction: VPSLLQ Z1, X1, K1, Z2
func simdVfpkv(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = v.Args[1].Reg()
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPCMPEQW Z26, Z30, K1, K4
func simdV2kk(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VPOPCNTB X14, K4, X16
func simdVkv(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[0])
	p.AddRestSourceReg(maskReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VROUNDPD $7, X2, X2
func simdV11Imm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VREDUCEPD $126, X1, K3, X31
func simdVkvImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VCMPPS $7, X2, X9, X2
func simdV21Imm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPINSRB $3, DX, X0, X0
func simdVgpvImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(v.Args[1].Reg())
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}
func simdVgpvImm(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	// within simdgen, the choice of intrinsic shape and the output
	// intruction format are linked.  In the case of VgpImm, there is
	// a difference in the intrinsic, but no difference in the
	// instruction, it is just like VgpvImm8.
	//
	// See also, simdVgpImm and simdVgpImm8
	return simdVgpvImm8(s, v)
}

// Example instruction: VPCMPD $1, Z1, Z2, K1
func simdV2kImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VPCMPD $1, Z1, Z2, K2, K1
func simdV2kkImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

func simdV2kvImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VFMADD213PD Z2, Z1, Z0
func simdV31ResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

func simdV31ResultInArg0Imm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST

	p.AddRestSourceReg(simdReg(v.Args[2]))
	p.AddRestSourceReg(simdReg(v.Args[1]))
	// p.AddRestSourceReg(x86.REG_K0)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// v31loadResultInArg0Imm8
// Example instruction:
// for (VPTERNLOGD128load {sym} [makeValAndOff(int32(int8(c)),off)]  x y ptr mem)
func simdV31loadResultInArg0Imm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())

	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()

	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[2].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)

	p.AddRestSourceReg(simdReg(v.Args[1]))
	return p
}

// Example instruction: VFMADD213PD Z2, Z1, K1, Z0
func simdV3kvResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(maskReg(v.Args[3]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

func simdVgpImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = v.Reg()
	return p
}

func simdVgpImm(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	// within simdgen, the choice of intrinsic shape and the output
	// intruction format are linked.  In the case of VgpImm, there is
	// a difference in the intrinsic, but no difference in the
	// instruction, it is just like VgpImm8.
	return simdVgpImm8(s, v)
}

// Currently unused
func simdV31(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Currently unused
func simdV3kv(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[3]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VRCP14PS (DI), K6, X22
func simdVkvload(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[0].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(maskReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSLLVD (DX), X7, X18
func simdV21load(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[1].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPDPWSSD (SI), X24, X18
func simdV31loadResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[2].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPDPWSSD (SI), X24, K1, X18
func simdV3kvloadResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[2].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(maskReg(v.Args[3]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSLLVD (SI), X1, K1, X2
func simdV2kvload(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[1].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPCMPEQD (SI), X1, K1
func simdV2kload(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[1].Reg()
	ssagen.AddAux(&p.From, v)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VCVTTPS2DQ (BX), X2
func simdV11load(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = v.Args[0].Reg()
	ssagen.AddAux(&p.From, v)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSHUFD $7, (BX), X11
func simdV11loadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[0].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPRORD $81, -15(R14), K7, Y1
func simdVkvloadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[0].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.AddRestSourceReg(maskReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSHLDD $82, 7(SI), Y21, Y3
func simdV21loadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VCMPPS $81, -7(DI), Y16, K3
func simdV2kloadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VCMPPS $81, -7(DI), Y16, K1, K3
func simdV2kkloadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = maskReg(v)
	return p
}

// Example instruction: VGF2P8AFFINEINVQB $64, -17(BP), X31, K3, X26
func simdV2kvloadImm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	sc := v.AuxValAndOff()
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = sc.Val64()
	m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()}
	ssagen.AddAux2(&m, v, sc.Off64())
	p.AddRestSource(m)
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(maskReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: SHA1NEXTE X2, X2
func simdV21ResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: SHA1RNDS4 $1, X2, X2
func simdV21ResultInArg0Imm8(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Offset = int64(v.AuxUInt8())
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: SHA256RNDS2 X0, X11, X2
func simdV31x0AtIn2ResultInArg0(s *ssagen.State, v *ssacore.Value) *obj.Prog {
	return simdV31ResultInArg0(s, v)
}

var blockJump = [...]struct {
	asm, invasm obj.As
}{
	block.BlockAMD64EQ:  {x86.AJEQ, x86.AJNE},
	block.BlockAMD64NE:  {x86.AJNE, x86.AJEQ},
	block.BlockAMD64LT:  {x86.AJLT, x86.AJGE},
	block.BlockAMD64GE:  {x86.AJGE, x86.AJLT},
	block.BlockAMD64LE:  {x86.AJLE, x86.AJGT},
	block.BlockAMD64GT:  {x86.AJGT, x86.AJLE},
	block.BlockAMD64OS:  {x86.AJOS, x86.AJOC},
	block.BlockAMD64OC:  {x86.AJOC, x86.AJOS},
	block.BlockAMD64ULT: {x86.AJCS, x86.AJCC},
	block.BlockAMD64UGE: {x86.AJCC, x86.AJCS},
	block.BlockAMD64UGT: {x86.AJHI, x86.AJLS},
	block.BlockAMD64ULE: {x86.AJLS, x86.AJHI},
	block.BlockAMD64ORD: {x86.AJPC, x86.AJPS},
	block.BlockAMD64NAN: {x86.AJPS, x86.AJPC},
}

var eqfJumps = [2][2]ssagen.IndexJump{
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPS, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPC, Index: 0}}, // next == b.Succs[1]
}
var nefJumps = [2][2]ssagen.IndexJump{
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPC, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPS, Index: 0}}, // next == b.Succs[1]
}

func ssaGenBlock(s *ssagen.State, b, next *ssacore.Block) {
	switch b.Kind {
	case block.BlockPlain, block.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case block.BlockExit, block.BlockRetJmp:
	case block.BlockRet:
		s.Prog(obj.ARET)

	case block.BlockAMD64EQF:
		s.CombJump(b, next, &eqfJumps)

	case block.BlockAMD64NEF:
		s.CombJump(b, next, &nefJumps)

	case block.BlockAMD64EQ, block.BlockAMD64NE,
		block.BlockAMD64LT, block.BlockAMD64GE,
		block.BlockAMD64LE, block.BlockAMD64GT,
		block.BlockAMD64OS, block.BlockAMD64OC,
		block.BlockAMD64ULT, block.BlockAMD64UGT,
		block.BlockAMD64ULE, block.BlockAMD64UGE:
		jmp := blockJump[b.Kind]
		switch next {
		case b.Succs[0].Block():
			s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssacore.BranchUnlikely {
				s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}

	case block.BlockAMD64JUMPTABLE:
		// JMP      *(TABLE)(INDEX*8)
		p := s.Prog(obj.AJMP)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = b.Controls[1].Reg()
		p.To.Index = b.Controls[0].Reg()
		p.To.Scale = 8
		// Save jump tables for later resolution of the target blocks.
		s.JumpTables = append(s.JumpTables, b)

	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}

func loadRegResult(s *ssagen.State, f *ssacore.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p := s.Prog(loadByRegWidth(reg, t.Size()))
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_AUTO
	p.From.Sym = n.Linksym()
	p.From.Offset = n.FrameOffset() + off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = reg
	return p
}

func spillArgReg(pp *objw.Progs, p *obj.Prog, f *ssacore.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p = pp.Append(p, storeByRegWidth(reg, t.Size()), obj.TYPE_REG, reg, 0, obj.TYPE_MEM, 0, n.FrameOffset()+off)
	p.To.Name = obj.NAME_PARAM
	p.To.Sym = n.Linksym()
	p.Pos = p.Pos.WithNotStmt()
	return p
}

// zero 16 bytes at reg+off.
func zero16(s *ssagen.State, reg int16, off int64) {
	//   MOVUPS  X15, off(ptrReg)
	p := s.Prog(x86.AMOVUPS)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_X15
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
}

// move 16 bytes from src+off to dst+off using temporary register tmp.
func move16(s *ssagen.State, src, dst, tmp int16, off int64) {
	//   MOVUPS  off(srcReg), tmpReg
	//   MOVUPS  tmpReg, off(dstReg)
	p := s.Prog(x86.AMOVUPS)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = src
	p.From.Offset = off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.AMOVUPS)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = tmp
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = dst
	p.To.Offset = off
}

// XXX maybe make this part of v.Reg?
// On the other hand, it is architecture-specific.
func simdReg(v *ssacore.Value) int16 {
	t := v.Type
	if !t.IsSIMD() {
		base.Fatalf("simdReg: not a simd type; v=%s, b=b%d, f=%s", v.LongString(), v.Block.ID, v.Block.Func.Name)
	}
	return simdRegBySize(v.Reg(), t.Size())
}

func simdRegBySize(reg int16, size int64) int16 {
	switch size {
	case 16:
		return reg
	case 32:
		return reg + (x86.REG_Y0 - x86.REG_X0)
	case 64:
		return reg + (x86.REG_Z0 - x86.REG_X0)
	}
	panic("simdRegBySize: bad size")
}

// XXX k mask
func maskReg(v *ssacore.Value) int16 {
	t := v.Type
	if !t.IsSIMD() {
		base.Fatalf("maskReg: not a simd type; v=%s, b=b%d, f=%s", v.LongString(), v.Block.ID, v.Block.Func.Name)
	}
	switch t.Size() {
	case 8:
		return v.Reg()
	}
	panic("unreachable")
}

// XXX k mask + vec
func simdOrMaskReg(v *ssacore.Value) int16 {
	t := v.Type
	if t.Size() <= 8 {
		return maskReg(v)
	}
	return simdReg(v)
}

// XXX this is used for shift operations only.
// regalloc will issue OpCopy with incorrect type, but the assigned
// register should be correct, and this function is merely checking
// the sanity of this part.
func simdCheckRegOnly(v *ssacore.Value, regStart, regEnd int16) int16 {
	if v.Reg() > regEnd || v.Reg() < regStart {
		panic("simdCheckRegOnly: not the desired register")
	}
	return v.Reg()
}
