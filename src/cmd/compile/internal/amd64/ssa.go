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
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {
	flive := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		flive = c.Type.IsFlags() || flive
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && (v.Op == ssa.OpAMD64MOVLconst || v.Op == ssa.OpAMD64MOVQconst) {
			// The "mark" is any non-nil Aux value.
			v.Aux = ssa.AuxMark
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

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	// Avoid partial register write
	if !t.IsFloat() {
		switch t.Size() {
		case 1:
			return x86.AMOVBLZX
		case 2:
			return x86.AMOVWLZX
		}
	}
	// Otherwise, there's no difference between load and store opcodes.
	return storeByType(t)
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	width := t.Size()
	if t.IsFloat() {
		switch width {
		case 4:
			return x86.AMOVSS
		case 8:
			return x86.AMOVSD
		}
	} else if t.IsSIMD() {
		return simdMov(width)
	} else {
		switch width {
		case 1:
			return x86.AMOVB
		case 2:
			return x86.AMOVW
		case 4:
			return x86.AMOVL
		case 8:
			return x86.AMOVQ
		case 16:
			return x86.AMOVUPS
		}
	}
	panic(fmt.Sprintf("bad store type %v", t))
}

// moveByType returns the reg->reg move instruction of the given type.
func moveByType(t *types.Type) obj.As {
	if t.IsFloat() {
		// Moving the whole sse2 register is faster
		// than moving just the correct low portion of it.
		// There is no xmm->xmm move with 1 byte opcode,
		// so use movups, which has 2 byte opcode.
		return x86.AMOVUPS
	} else if t.IsSIMD() {
		return simdMov(t.Size())
	} else {
		switch t.Size() {
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
			return x86.AMOVUPS // int128s are in SSE registers
		default:
			panic(fmt.Sprintf("bad int register width %d:%v", t.Size(), t))
		}
	}
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
func memIdx(a *obj.Addr, v *ssa.Value) {
	r, i := v.Args[0].Reg(), v.Args[1].Reg()
	a.Type = obj.TYPE_MEM
	a.Scale = v.Op.Scale()
	if a.Scale == 1 && i == x86.REG_SP {
		r, i = i, r
	}
	a.Reg = r
	a.Index = i
}

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + LEAQ,
// See runtime/mkduff.go.
func duffStart(size int64) int64 {
	x, _ := duff(size)
	return x
}
func duffAdj(size int64) int64 {
	_, x := duff(size)
	return x
}

// duff returns the offset (from duffzero, in bytes) and pointer adjust (in bytes)
// required to use the duffzero mechanism for a block of the given size.
func duff(size int64) (int64, int64) {
	if size < 32 || size > 1024 || size%dzClearStep != 0 {
		panic("bad duffzero size")
	}
	steps := size / dzClearStep
	blocks := steps / dzBlockLen
	steps %= dzBlockLen
	off := dzBlockSize * (dzBlocks - blocks)
	var adj int64
	if steps != 0 {
		off -= dzLeaqSize
		off -= dzMovSize * steps
		adj -= dzClearStep * (dzBlockLen - steps)
	}
	return off, adj
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

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssa.OpAMD64VFMADD231SD, ssa.OpAMD64VFMADD231SS:
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: v.Args[2].Reg()}
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: v.Reg()}
		p.AddRestSourceReg(v.Args[1].Reg())
	case ssa.OpAMD64ADDQ, ssa.OpAMD64ADDL:
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
			if v.Op == ssa.OpAMD64ADDQ {
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
	case ssa.OpAMD64SUBQ, ssa.OpAMD64SUBL,
		ssa.OpAMD64MULQ, ssa.OpAMD64MULL,
		ssa.OpAMD64ANDQ, ssa.OpAMD64ANDL,
		ssa.OpAMD64ORQ, ssa.OpAMD64ORL,
		ssa.OpAMD64XORQ, ssa.OpAMD64XORL,
		ssa.OpAMD64SHLQ, ssa.OpAMD64SHLL,
		ssa.OpAMD64SHRQ, ssa.OpAMD64SHRL, ssa.OpAMD64SHRW, ssa.OpAMD64SHRB,
		ssa.OpAMD64SARQ, ssa.OpAMD64SARL, ssa.OpAMD64SARW, ssa.OpAMD64SARB,
		ssa.OpAMD64ROLQ, ssa.OpAMD64ROLL, ssa.OpAMD64ROLW, ssa.OpAMD64ROLB,
		ssa.OpAMD64RORQ, ssa.OpAMD64RORL, ssa.OpAMD64RORW, ssa.OpAMD64RORB,
		ssa.OpAMD64ADDSS, ssa.OpAMD64ADDSD, ssa.OpAMD64SUBSS, ssa.OpAMD64SUBSD,
		ssa.OpAMD64MULSS, ssa.OpAMD64MULSD, ssa.OpAMD64DIVSS, ssa.OpAMD64DIVSD,
		ssa.OpAMD64MINSS, ssa.OpAMD64MINSD,
		ssa.OpAMD64POR, ssa.OpAMD64PXOR,
		ssa.OpAMD64BTSL, ssa.OpAMD64BTSQ,
		ssa.OpAMD64BTCL, ssa.OpAMD64BTCQ,
		ssa.OpAMD64BTRL, ssa.OpAMD64BTRQ,
		ssa.OpAMD64PCMPEQB, ssa.OpAMD64PSIGNB,
		ssa.OpAMD64PUNPCKLBW:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())

	case ssa.OpAMD64PSHUFLW:
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

	case ssa.OpAMD64PSHUFBbroadcast:
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

	case ssa.OpAMD64SHRDQ, ssa.OpAMD64SHLDQ:
		p := s.Prog(v.Op.Asm())
		lo, hi, bits := v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg()
		p.From.Type = obj.TYPE_REG
		p.From.Reg = bits
		p.To.Type = obj.TYPE_REG
		p.To.Reg = lo
		p.AddRestSourceReg(hi)

	case ssa.OpAMD64BLSIQ, ssa.OpAMD64BLSIL,
		ssa.OpAMD64BLSMSKQ, ssa.OpAMD64BLSMSKL,
		ssa.OpAMD64BLSRQ, ssa.OpAMD64BLSRL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		switch v.Op {
		case ssa.OpAMD64BLSRQ, ssa.OpAMD64BLSRL:
			p.To.Reg = v.Reg0()
		default:
			p.To.Reg = v.Reg()
		}

	case ssa.OpAMD64ANDNQ, ssa.OpAMD64ANDNL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.AddRestSourceReg(v.Args[1].Reg())

	case ssa.OpAMD64SARXL, ssa.OpAMD64SARXQ,
		ssa.OpAMD64SHLXL, ssa.OpAMD64SHLXQ,
		ssa.OpAMD64SHRXL, ssa.OpAMD64SHRXQ:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		p.AddRestSourceReg(v.Args[0].Reg())

	case ssa.OpAMD64SHLXLload, ssa.OpAMD64SHLXQload,
		ssa.OpAMD64SHRXLload, ssa.OpAMD64SHRXQload,
		ssa.OpAMD64SARXLload, ssa.OpAMD64SARXQload:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		m := obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[0].Reg()}
		ssagen.AddAux(&m, v)
		p.AddRestSource(m)

	case ssa.OpAMD64SHLXLloadidx1, ssa.OpAMD64SHLXLloadidx4, ssa.OpAMD64SHLXLloadidx8,
		ssa.OpAMD64SHRXLloadidx1, ssa.OpAMD64SHRXLloadidx4, ssa.OpAMD64SHRXLloadidx8,
		ssa.OpAMD64SARXLloadidx1, ssa.OpAMD64SARXLloadidx4, ssa.OpAMD64SARXLloadidx8,
		ssa.OpAMD64SHLXQloadidx1, ssa.OpAMD64SHLXQloadidx8,
		ssa.OpAMD64SHRXQloadidx1, ssa.OpAMD64SHRXQloadidx8,
		ssa.OpAMD64SARXQloadidx1, ssa.OpAMD64SARXQloadidx8:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[2].Reg())
		m := obj.Addr{Type: obj.TYPE_MEM}
		memIdx(&m, v)
		ssagen.AddAux(&m, v)
		p.AddRestSource(m)

	case ssa.OpAMD64DIVQU, ssa.OpAMD64DIVLU, ssa.OpAMD64DIVWU:
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

	case ssa.OpAMD64DIVQ, ssa.OpAMD64DIVL, ssa.OpAMD64DIVW:
		// Arg[0] (the dividend) is in AX.
		// Arg[1] (the divisor) can be in any other register.
		// Result[0] (the quotient) is in AX.
		// Result[1] (the remainder) is in DX.
		r := v.Args[1].Reg()

		var opCMP, opNEG, opSXD obj.As
		switch v.Op {
		case ssa.OpAMD64DIVQ:
			opCMP, opNEG, opSXD = x86.ACMPQ, x86.ANEGQ, x86.ACQO
		case ssa.OpAMD64DIVL:
			opCMP, opNEG, opSXD = x86.ACMPL, x86.ANEGL, x86.ACDQ
		case ssa.OpAMD64DIVW:
			opCMP, opNEG, opSXD = x86.ACMPW, x86.ANEGW, x86.ACWD
		}

		// CPU faults upon signed overflow, which occurs when the most
		// negative int is divided by -1. Handle divide by -1 as a special case.
		var j1, j2 *obj.Prog
		if ssa.DivisionNeedsFixUp(v) {
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

	case ssa.OpAMD64HMULQ, ssa.OpAMD64HMULL, ssa.OpAMD64HMULQU, ssa.OpAMD64HMULLU:
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

	case ssa.OpAMD64MULQU, ssa.OpAMD64MULLU:
		// Arg[0] is already in AX as it's the only register we allow
		// results lo in AX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssa.OpAMD64MULQU2:
		// Arg[0] is already in AX as it's the only register we allow
		// results hi in DX, lo in AX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssa.OpAMD64DIVQU2:
		// Arg[0], Arg[1] are already in Dx, AX, as they're the only registers we allow
		// results q in AX, r in DX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()

	case ssa.OpAMD64AVGQU:
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

	case ssa.OpAMD64ADDQcarry, ssa.OpAMD64ADCQ:
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

	case ssa.OpAMD64SUBQborrow, ssa.OpAMD64SBBQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssa.OpAMD64ADDQconstcarry, ssa.OpAMD64ADCQconst, ssa.OpAMD64SUBQconstborrow, ssa.OpAMD64SBBQconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssa.OpAMD64ADDQconst, ssa.OpAMD64ADDLconst:
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
				if v.Op == ssa.OpAMD64ADDQconst {
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
				if v.Op == ssa.OpAMD64ADDQconst {
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
				if v.Op == ssa.OpAMD64ADDQconst {
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
		if v.Op == ssa.OpAMD64ADDQconst {
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

	case ssa.OpAMD64CMOVQEQ, ssa.OpAMD64CMOVLEQ, ssa.OpAMD64CMOVWEQ,
		ssa.OpAMD64CMOVQLT, ssa.OpAMD64CMOVLLT, ssa.OpAMD64CMOVWLT,
		ssa.OpAMD64CMOVQNE, ssa.OpAMD64CMOVLNE, ssa.OpAMD64CMOVWNE,
		ssa.OpAMD64CMOVQGT, ssa.OpAMD64CMOVLGT, ssa.OpAMD64CMOVWGT,
		ssa.OpAMD64CMOVQLE, ssa.OpAMD64CMOVLLE, ssa.OpAMD64CMOVWLE,
		ssa.OpAMD64CMOVQGE, ssa.OpAMD64CMOVLGE, ssa.OpAMD64CMOVWGE,
		ssa.OpAMD64CMOVQHI, ssa.OpAMD64CMOVLHI, ssa.OpAMD64CMOVWHI,
		ssa.OpAMD64CMOVQLS, ssa.OpAMD64CMOVLLS, ssa.OpAMD64CMOVWLS,
		ssa.OpAMD64CMOVQCC, ssa.OpAMD64CMOVLCC, ssa.OpAMD64CMOVWCC,
		ssa.OpAMD64CMOVQCS, ssa.OpAMD64CMOVLCS, ssa.OpAMD64CMOVWCS,
		ssa.OpAMD64CMOVQGTF, ssa.OpAMD64CMOVLGTF, ssa.OpAMD64CMOVWGTF,
		ssa.OpAMD64CMOVQGEF, ssa.OpAMD64CMOVLGEF, ssa.OpAMD64CMOVWGEF:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpAMD64CMOVQNEF, ssa.OpAMD64CMOVLNEF, ssa.OpAMD64CMOVWNEF:
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
		if v.Op == ssa.OpAMD64CMOVQNEF {
			q = s.Prog(x86.ACMOVQPS)
		} else if v.Op == ssa.OpAMD64CMOVLNEF {
			q = s.Prog(x86.ACMOVLPS)
		} else {
			q = s.Prog(x86.ACMOVWPS)
		}
		q.From.Type = obj.TYPE_REG
		q.From.Reg = v.Args[1].Reg()
		q.To.Type = obj.TYPE_REG
		q.To.Reg = v.Reg()

	case ssa.OpAMD64CMOVQEQF, ssa.OpAMD64CMOVLEQF, ssa.OpAMD64CMOVWEQF:
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
		opregreg(s, moveByType(v.Type), t, v.Args[1].Reg())

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = t
		var q *obj.Prog
		if v.Op == ssa.OpAMD64CMOVQEQF {
			q = s.Prog(x86.ACMOVQPC)
		} else if v.Op == ssa.OpAMD64CMOVLEQF {
			q = s.Prog(x86.ACMOVLPC)
		} else {
			q = s.Prog(x86.ACMOVWPC)
		}
		q.From.Type = obj.TYPE_REG
		q.From.Reg = t
		q.To.Type = obj.TYPE_REG
		q.To.Reg = v.Reg()

	case ssa.OpAMD64MULQconst, ssa.OpAMD64MULLconst:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.AddRestSourceReg(v.Args[0].Reg())

	case ssa.OpAMD64ANDQconst:
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

	case ssa.OpAMD64SUBQconst, ssa.OpAMD64SUBLconst,
		ssa.OpAMD64ANDLconst,
		ssa.OpAMD64ORQconst, ssa.OpAMD64ORLconst,
		ssa.OpAMD64XORQconst, ssa.OpAMD64XORLconst,
		ssa.OpAMD64SHLQconst, ssa.OpAMD64SHLLconst,
		ssa.OpAMD64SHRQconst, ssa.OpAMD64SHRLconst, ssa.OpAMD64SHRWconst, ssa.OpAMD64SHRBconst,
		ssa.OpAMD64SARQconst, ssa.OpAMD64SARLconst, ssa.OpAMD64SARWconst, ssa.OpAMD64SARBconst,
		ssa.OpAMD64ROLQconst, ssa.OpAMD64ROLLconst, ssa.OpAMD64ROLWconst, ssa.OpAMD64ROLBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64SBBQcarrymask, ssa.OpAMD64SBBLcarrymask:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64LEAQ1, ssa.OpAMD64LEAQ2, ssa.OpAMD64LEAQ4, ssa.OpAMD64LEAQ8,
		ssa.OpAMD64LEAL1, ssa.OpAMD64LEAL2, ssa.OpAMD64LEAL4, ssa.OpAMD64LEAL8,
		ssa.OpAMD64LEAW1, ssa.OpAMD64LEAW2, ssa.OpAMD64LEAW4, ssa.OpAMD64LEAW8:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		o := v.Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = o
		if v.AuxInt != 0 && v.Aux == nil {
			// Emit an additional LEA to add the displacement instead of creating a slow 3 operand LEA.
			switch v.Op {
			case ssa.OpAMD64LEAQ1, ssa.OpAMD64LEAQ2, ssa.OpAMD64LEAQ4, ssa.OpAMD64LEAQ8:
				p = s.Prog(x86.ALEAQ)
			case ssa.OpAMD64LEAL1, ssa.OpAMD64LEAL2, ssa.OpAMD64LEAL4, ssa.OpAMD64LEAL8:
				p = s.Prog(x86.ALEAL)
			case ssa.OpAMD64LEAW1, ssa.OpAMD64LEAW2, ssa.OpAMD64LEAW4, ssa.OpAMD64LEAW8:
				p = s.Prog(x86.ALEAW)
			}
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = o
			p.To.Type = obj.TYPE_REG
			p.To.Reg = o
		}
		ssagen.AddAux(&p.From, v)
	case ssa.OpAMD64LEAQ, ssa.OpAMD64LEAL, ssa.OpAMD64LEAW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64CMPQ, ssa.OpAMD64CMPL, ssa.OpAMD64CMPW, ssa.OpAMD64CMPB,
		ssa.OpAMD64TESTQ, ssa.OpAMD64TESTL, ssa.OpAMD64TESTW, ssa.OpAMD64TESTB,
		ssa.OpAMD64BTL, ssa.OpAMD64BTQ:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssa.OpAMD64UCOMISS, ssa.OpAMD64UCOMISD:
		// Go assembler has swapped operands for UCOMISx relative to CMP,
		// must account for that right here.
		opregreg(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg())
	case ssa.OpAMD64CMPQconst, ssa.OpAMD64CMPLconst, ssa.OpAMD64CMPWconst, ssa.OpAMD64CMPBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.OpAMD64BTLconst, ssa.OpAMD64BTQconst,
		ssa.OpAMD64TESTQconst, ssa.OpAMD64TESTLconst, ssa.OpAMD64TESTWconst, ssa.OpAMD64TESTBconst,
		ssa.OpAMD64BTSQconst,
		ssa.OpAMD64BTCQconst,
		ssa.OpAMD64BTRQconst:
		op := v.Op
		if op == ssa.OpAMD64BTQconst && v.AuxInt < 32 {
			// Emit 32-bit version because it's shorter
			op = ssa.OpAMD64BTLconst
		}
		p := s.Prog(op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
	case ssa.OpAMD64CMPQload, ssa.OpAMD64CMPLload, ssa.OpAMD64CMPWload, ssa.OpAMD64CMPBload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[1].Reg()
	case ssa.OpAMD64CMPQconstload, ssa.OpAMD64CMPLconstload, ssa.OpAMD64CMPWconstload, ssa.OpAMD64CMPBconstload:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.From, v, sc.Off64())
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = sc.Val64()
	case ssa.OpAMD64CMPQloadidx8, ssa.OpAMD64CMPQloadidx1, ssa.OpAMD64CMPLloadidx4, ssa.OpAMD64CMPLloadidx1, ssa.OpAMD64CMPWloadidx2, ssa.OpAMD64CMPWloadidx1, ssa.OpAMD64CMPBloadidx1:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[2].Reg()
	case ssa.OpAMD64CMPQconstloadidx8, ssa.OpAMD64CMPQconstloadidx1, ssa.OpAMD64CMPLconstloadidx4, ssa.OpAMD64CMPLconstloadidx1, ssa.OpAMD64CMPWconstloadidx2, ssa.OpAMD64CMPWconstloadidx1, ssa.OpAMD64CMPBconstloadidx1:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux2(&p.From, v, sc.Off64())
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = sc.Val64()
	case ssa.OpAMD64MOVLconst, ssa.OpAMD64MOVQconst:
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
	case ssa.OpAMD64MOVSSconst, ssa.OpAMD64MOVSDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpAMD64MOVQload, ssa.OpAMD64MOVLload, ssa.OpAMD64MOVWload, ssa.OpAMD64MOVBload, ssa.OpAMD64MOVOload,
		ssa.OpAMD64MOVSSload, ssa.OpAMD64MOVSDload, ssa.OpAMD64MOVBQSXload, ssa.OpAMD64MOVWQSXload, ssa.OpAMD64MOVLQSXload,
		ssa.OpAMD64MOVBEQload, ssa.OpAMD64MOVBELload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64MOVBloadidx1, ssa.OpAMD64MOVWloadidx1, ssa.OpAMD64MOVLloadidx1, ssa.OpAMD64MOVQloadidx1, ssa.OpAMD64MOVSSloadidx1, ssa.OpAMD64MOVSDloadidx1,
		ssa.OpAMD64MOVQloadidx8, ssa.OpAMD64MOVSDloadidx8, ssa.OpAMD64MOVLloadidx8, ssa.OpAMD64MOVLloadidx4, ssa.OpAMD64MOVSSloadidx4, ssa.OpAMD64MOVWloadidx2,
		ssa.OpAMD64MOVBELloadidx1, ssa.OpAMD64MOVBELloadidx4, ssa.OpAMD64MOVBELloadidx8, ssa.OpAMD64MOVBEQloadidx1, ssa.OpAMD64MOVBEQloadidx8:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.From, v)
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64MOVQstore, ssa.OpAMD64MOVSSstore, ssa.OpAMD64MOVSDstore, ssa.OpAMD64MOVLstore, ssa.OpAMD64MOVWstore, ssa.OpAMD64MOVBstore, ssa.OpAMD64MOVOstore,
		ssa.OpAMD64ADDQmodify, ssa.OpAMD64SUBQmodify, ssa.OpAMD64ANDQmodify, ssa.OpAMD64ORQmodify, ssa.OpAMD64XORQmodify,
		ssa.OpAMD64ADDLmodify, ssa.OpAMD64SUBLmodify, ssa.OpAMD64ANDLmodify, ssa.OpAMD64ORLmodify, ssa.OpAMD64XORLmodify,
		ssa.OpAMD64MOVBEQstore, ssa.OpAMD64MOVBELstore, ssa.OpAMD64MOVBEWstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpAMD64MOVBstoreidx1, ssa.OpAMD64MOVWstoreidx1, ssa.OpAMD64MOVLstoreidx1, ssa.OpAMD64MOVQstoreidx1, ssa.OpAMD64MOVSSstoreidx1, ssa.OpAMD64MOVSDstoreidx1,
		ssa.OpAMD64MOVQstoreidx8, ssa.OpAMD64MOVSDstoreidx8, ssa.OpAMD64MOVLstoreidx8, ssa.OpAMD64MOVSSstoreidx4, ssa.OpAMD64MOVLstoreidx4, ssa.OpAMD64MOVWstoreidx2,
		ssa.OpAMD64ADDLmodifyidx1, ssa.OpAMD64ADDLmodifyidx4, ssa.OpAMD64ADDLmodifyidx8, ssa.OpAMD64ADDQmodifyidx1, ssa.OpAMD64ADDQmodifyidx8,
		ssa.OpAMD64SUBLmodifyidx1, ssa.OpAMD64SUBLmodifyidx4, ssa.OpAMD64SUBLmodifyidx8, ssa.OpAMD64SUBQmodifyidx1, ssa.OpAMD64SUBQmodifyidx8,
		ssa.OpAMD64ANDLmodifyidx1, ssa.OpAMD64ANDLmodifyidx4, ssa.OpAMD64ANDLmodifyidx8, ssa.OpAMD64ANDQmodifyidx1, ssa.OpAMD64ANDQmodifyidx8,
		ssa.OpAMD64ORLmodifyidx1, ssa.OpAMD64ORLmodifyidx4, ssa.OpAMD64ORLmodifyidx8, ssa.OpAMD64ORQmodifyidx1, ssa.OpAMD64ORQmodifyidx8,
		ssa.OpAMD64XORLmodifyidx1, ssa.OpAMD64XORLmodifyidx4, ssa.OpAMD64XORLmodifyidx8, ssa.OpAMD64XORQmodifyidx1, ssa.OpAMD64XORQmodifyidx8,
		ssa.OpAMD64MOVBEWstoreidx1, ssa.OpAMD64MOVBEWstoreidx2, ssa.OpAMD64MOVBELstoreidx1, ssa.OpAMD64MOVBELstoreidx4, ssa.OpAMD64MOVBELstoreidx8, ssa.OpAMD64MOVBEQstoreidx1, ssa.OpAMD64MOVBEQstoreidx8:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		memIdx(&p.To, v)
		ssagen.AddAux(&p.To, v)
	case ssa.OpAMD64ADDQconstmodify, ssa.OpAMD64ADDLconstmodify:
		sc := v.AuxValAndOff()
		off := sc.Off64()
		val := sc.Val()
		if val == 1 || val == -1 {
			var asm obj.As
			if v.Op == ssa.OpAMD64ADDQconstmodify {
				if val == 1 {
					asm = x86.AINCQ
				} else {
					asm = x86.ADECQ
				}
			} else {
				if val == 1 {
					asm = x86.AINCL
				} else {
					asm = x86.ADECL
				}
			}
			p := s.Prog(asm)
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = v.Args[0].Reg()
			ssagen.AddAux2(&p.To, v, off)
			break
		}
		fallthrough
	case ssa.OpAMD64ANDQconstmodify, ssa.OpAMD64ANDLconstmodify, ssa.OpAMD64ORQconstmodify, ssa.OpAMD64ORLconstmodify,
		ssa.OpAMD64XORQconstmodify, ssa.OpAMD64XORLconstmodify,
		ssa.OpAMD64BTSQconstmodify, ssa.OpAMD64BTRQconstmodify, ssa.OpAMD64BTCQconstmodify:
		sc := v.AuxValAndOff()
		off := sc.Off64()
		val := sc.Val64()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = val
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, off)

	case ssa.OpAMD64MOVQstoreconst, ssa.OpAMD64MOVLstoreconst, ssa.OpAMD64MOVWstoreconst, ssa.OpAMD64MOVBstoreconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.OpAMD64MOVOstoreconst:
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

	case ssa.OpAMD64MOVQstoreconstidx1, ssa.OpAMD64MOVQstoreconstidx8, ssa.OpAMD64MOVLstoreconstidx1, ssa.OpAMD64MOVLstoreconstidx4, ssa.OpAMD64MOVWstoreconstidx1, ssa.OpAMD64MOVWstoreconstidx2, ssa.OpAMD64MOVBstoreconstidx1,
		ssa.OpAMD64ADDLconstmodifyidx1, ssa.OpAMD64ADDLconstmodifyidx4, ssa.OpAMD64ADDLconstmodifyidx8, ssa.OpAMD64ADDQconstmodifyidx1, ssa.OpAMD64ADDQconstmodifyidx8,
		ssa.OpAMD64ANDLconstmodifyidx1, ssa.OpAMD64ANDLconstmodifyidx4, ssa.OpAMD64ANDLconstmodifyidx8, ssa.OpAMD64ANDQconstmodifyidx1, ssa.OpAMD64ANDQconstmodifyidx8,
		ssa.OpAMD64ORLconstmodifyidx1, ssa.OpAMD64ORLconstmodifyidx4, ssa.OpAMD64ORLconstmodifyidx8, ssa.OpAMD64ORQconstmodifyidx1, ssa.OpAMD64ORQconstmodifyidx8,
		ssa.OpAMD64XORLconstmodifyidx1, ssa.OpAMD64XORLconstmodifyidx4, ssa.OpAMD64XORLconstmodifyidx8, ssa.OpAMD64XORQconstmodifyidx1, ssa.OpAMD64XORQconstmodifyidx8:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
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
		memIdx(&p.To, v)
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.OpAMD64MOVLQSX, ssa.OpAMD64MOVWQSX, ssa.OpAMD64MOVBQSX, ssa.OpAMD64MOVLQZX, ssa.OpAMD64MOVWQZX, ssa.OpAMD64MOVBQZX,
		ssa.OpAMD64CVTTSS2SL, ssa.OpAMD64CVTTSD2SL, ssa.OpAMD64CVTTSS2SQ, ssa.OpAMD64CVTTSD2SQ,
		ssa.OpAMD64CVTSS2SD, ssa.OpAMD64CVTSD2SS, ssa.OpAMD64VPBROADCASTB, ssa.OpAMD64PMOVMSKB:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssa.OpAMD64CVTSL2SD, ssa.OpAMD64CVTSQ2SD, ssa.OpAMD64CVTSQ2SS, ssa.OpAMD64CVTSL2SS:
		r := v.Reg()
		// Break false dependency on destination register.
		opregreg(s, x86.AXORPS, r, r)
		opregreg(s, v.Op.Asm(), r, v.Args[0].Reg())
	case ssa.OpAMD64MOVQi2f, ssa.OpAMD64MOVQf2i, ssa.OpAMD64MOVLi2f, ssa.OpAMD64MOVLf2i:
		var p *obj.Prog
		switch v.Op {
		case ssa.OpAMD64MOVQi2f, ssa.OpAMD64MOVQf2i:
			p = s.Prog(x86.AMOVQ)
		case ssa.OpAMD64MOVLi2f, ssa.OpAMD64MOVLf2i:
			p = s.Prog(x86.AMOVL)
		}
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64ADDQload, ssa.OpAMD64ADDLload, ssa.OpAMD64SUBQload, ssa.OpAMD64SUBLload,
		ssa.OpAMD64ANDQload, ssa.OpAMD64ANDLload, ssa.OpAMD64ORQload, ssa.OpAMD64ORLload,
		ssa.OpAMD64XORQload, ssa.OpAMD64XORLload, ssa.OpAMD64ADDSDload, ssa.OpAMD64ADDSSload,
		ssa.OpAMD64SUBSDload, ssa.OpAMD64SUBSSload, ssa.OpAMD64MULSDload, ssa.OpAMD64MULSSload,
		ssa.OpAMD64DIVSDload, ssa.OpAMD64DIVSSload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpAMD64ADDLloadidx1, ssa.OpAMD64ADDLloadidx4, ssa.OpAMD64ADDLloadidx8, ssa.OpAMD64ADDQloadidx1, ssa.OpAMD64ADDQloadidx8,
		ssa.OpAMD64SUBLloadidx1, ssa.OpAMD64SUBLloadidx4, ssa.OpAMD64SUBLloadidx8, ssa.OpAMD64SUBQloadidx1, ssa.OpAMD64SUBQloadidx8,
		ssa.OpAMD64ANDLloadidx1, ssa.OpAMD64ANDLloadidx4, ssa.OpAMD64ANDLloadidx8, ssa.OpAMD64ANDQloadidx1, ssa.OpAMD64ANDQloadidx8,
		ssa.OpAMD64ORLloadidx1, ssa.OpAMD64ORLloadidx4, ssa.OpAMD64ORLloadidx8, ssa.OpAMD64ORQloadidx1, ssa.OpAMD64ORQloadidx8,
		ssa.OpAMD64XORLloadidx1, ssa.OpAMD64XORLloadidx4, ssa.OpAMD64XORLloadidx8, ssa.OpAMD64XORQloadidx1, ssa.OpAMD64XORQloadidx8,
		ssa.OpAMD64ADDSSloadidx1, ssa.OpAMD64ADDSSloadidx4, ssa.OpAMD64ADDSDloadidx1, ssa.OpAMD64ADDSDloadidx8,
		ssa.OpAMD64SUBSSloadidx1, ssa.OpAMD64SUBSSloadidx4, ssa.OpAMD64SUBSDloadidx1, ssa.OpAMD64SUBSDloadidx8,
		ssa.OpAMD64MULSSloadidx1, ssa.OpAMD64MULSSloadidx4, ssa.OpAMD64MULSDloadidx1, ssa.OpAMD64MULSDloadidx8,
		ssa.OpAMD64DIVSSloadidx1, ssa.OpAMD64DIVSSloadidx4, ssa.OpAMD64DIVSDloadidx1, ssa.OpAMD64DIVSDloadidx8:
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
	case ssa.OpAMD64DUFFZERO:
		if s.ABI != obj.ABIInternal {
			// zero X15 manually
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
		}
		off := duffStart(v.AuxInt)
		adj := duffAdj(v.AuxInt)
		var p *obj.Prog
		if adj != 0 {
			p = s.Prog(x86.ALEAQ)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = adj
			p.From.Reg = x86.REG_DI
			p.To.Type = obj.TYPE_REG
			p.To.Reg = x86.REG_DI
		}
		p = s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = off
	case ssa.OpAMD64DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = ir.Syms.Duffcopy
		if v.AuxInt%16 != 0 {
			v.Fatalf("bad DUFFCOPY AuxInt %v", v.AuxInt)
		}
		p.To.Offset = 14 * (64 - v.AuxInt/16)
		// 14 and 64 are magic constants.  14 is the number of bytes to encode:
		//	MOVUPS	(SI), X0
		//	ADDQ	$16, SI
		//	MOVUPS	X0, (DI)
		//	ADDQ	$16, DI
		// and 64 is the number of such blocks. See src/runtime/duff_amd64.s:duffcopy.

	case ssa.OpCopy: // TODO: use MOVQreg for reg->reg copies instead of OpCopy?
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if v.Type.IsSIMD() {
			x = simdReg(v.Args[0])
			y = simdReg(v)
		}
		if x != y {
			opregreg(s, moveByType(v.Type), y, x)
		}
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(loadByType(v.Type))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		r := v.Reg()
		if v.Type.IsSIMD() {
			r = simdReg(v)
		}
		p.To.Reg = r

	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		if v.Type.IsSIMD() {
			r = simdReg(v.Args[0])
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		ssagen.AddrAuto(&p.To, v)
	case ssa.OpAMD64LoweredHasCPUFeature:
		p := s.Prog(x86.AMOVBLZX)
		p.From.Type = obj.TYPE_MEM
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpArgIntReg, ssa.OpArgFloatReg:
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		for _, ap := range v.Block.Func.RegArgs {
			// Pass the spill/unspill information along to the assembler, offset by size of return PC pushed on stack.
			addr := ssagen.SpillSlotAddr(ap, x86.REG_SP, v.Block.Func.Config.PtrSize)
			s.FuncInfo().AddSpill(
				obj.RegSpill{Reg: ap.Reg, Addr: addr, Unspill: loadByType(ap.Type), Spill: storeByType(ap.Type)})
		}
		v.Block.Func.RegArgs = nil
		ssagen.CheckArgReg(v)
	case ssa.OpAMD64LoweredGetClosurePtr:
		// Closure pointer is DX.
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssa.OpAMD64LoweredGetG:
		if s.ABI == obj.ABIInternal {
			v.Fatalf("LoweredGetG should not appear in ABIInternal")
		}
		r := v.Reg()
		getgFromTLS(s, r)
	case ssa.OpAMD64CALLstatic, ssa.OpAMD64CALLtail:
		if s.ABI == obj.ABI0 && v.Aux.(*ssa.AuxCall).Fn.ABI() == obj.ABIInternal {
			// zeroing X15 when entering ABIInternal from ABI0
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
			// set G register from TLS
			getgFromTLS(s, x86.REG_R14)
		}
		if v.Op == ssa.OpAMD64CALLtail {
			s.TailCall(v)
			break
		}
		s.Call(v)
		if s.ABI == obj.ABIInternal && v.Aux.(*ssa.AuxCall).Fn.ABI() == obj.ABI0 {
			// zeroing X15 when entering ABIInternal from ABI0
			opregreg(s, x86.AXORPS, x86.REG_X15, x86.REG_X15)
			// set G register from TLS
			getgFromTLS(s, x86.REG_R14)
		}
	case ssa.OpAMD64CALLclosure, ssa.OpAMD64CALLinter:
		s.Call(v)

	case ssa.OpAMD64LoweredGetCallerPC:
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = -8 // PC is stored 8 bytes below first parameter.
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpAMD64LoweredGetCallerSP:
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

	case ssa.OpAMD64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssa.OpAMD64LoweredPanicBoundsA, ssa.OpAMD64LoweredPanicBoundsB, ssa.OpAMD64LoweredPanicBoundsC:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ssagen.BoundsCheckFunc[v.AuxInt]
		s.UseArgs(int64(2 * types.PtrSize)) // space used in callee args area by assembly stubs

	case ssa.OpAMD64NEGQ, ssa.OpAMD64NEGL,
		ssa.OpAMD64BSWAPQ, ssa.OpAMD64BSWAPL,
		ssa.OpAMD64NOTQ, ssa.OpAMD64NOTL:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpAMD64NEGLflags:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssa.OpAMD64ADDQconstflags, ssa.OpAMD64ADDLconstflags:
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

	case ssa.OpAMD64BSFQ, ssa.OpAMD64BSRQ, ssa.OpAMD64BSFL, ssa.OpAMD64BSRL, ssa.OpAMD64SQRTSD, ssa.OpAMD64SQRTSS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		switch v.Op {
		case ssa.OpAMD64BSFQ, ssa.OpAMD64BSRQ:
			p.To.Reg = v.Reg0()
		case ssa.OpAMD64BSFL, ssa.OpAMD64BSRL, ssa.OpAMD64SQRTSD, ssa.OpAMD64SQRTSS:
			p.To.Reg = v.Reg()
		}
	case ssa.OpAMD64LoweredRound32F, ssa.OpAMD64LoweredRound64F:
		// input is already rounded
	case ssa.OpAMD64ROUNDSD:
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
	case ssa.OpAMD64POPCNTQ, ssa.OpAMD64POPCNTL,
		ssa.OpAMD64TZCNTQ, ssa.OpAMD64TZCNTL,
		ssa.OpAMD64LZCNTQ, ssa.OpAMD64LZCNTL:
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

	case ssa.OpAMD64SETEQ, ssa.OpAMD64SETNE,
		ssa.OpAMD64SETL, ssa.OpAMD64SETLE,
		ssa.OpAMD64SETG, ssa.OpAMD64SETGE,
		ssa.OpAMD64SETGF, ssa.OpAMD64SETGEF,
		ssa.OpAMD64SETB, ssa.OpAMD64SETBE,
		ssa.OpAMD64SETORD, ssa.OpAMD64SETNAN,
		ssa.OpAMD64SETA, ssa.OpAMD64SETAE,
		ssa.OpAMD64SETO:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpAMD64SETEQstore, ssa.OpAMD64SETNEstore,
		ssa.OpAMD64SETLstore, ssa.OpAMD64SETLEstore,
		ssa.OpAMD64SETGstore, ssa.OpAMD64SETGEstore,
		ssa.OpAMD64SETBstore, ssa.OpAMD64SETBEstore,
		ssa.OpAMD64SETAstore, ssa.OpAMD64SETAEstore:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)

	case ssa.OpAMD64SETEQstoreidx1, ssa.OpAMD64SETNEstoreidx1,
		ssa.OpAMD64SETLstoreidx1, ssa.OpAMD64SETLEstoreidx1,
		ssa.OpAMD64SETGstoreidx1, ssa.OpAMD64SETGEstoreidx1,
		ssa.OpAMD64SETBstoreidx1, ssa.OpAMD64SETBEstoreidx1,
		ssa.OpAMD64SETAstoreidx1, ssa.OpAMD64SETAEstoreidx1:
		p := s.Prog(v.Op.Asm())
		memIdx(&p.To, v)
		ssagen.AddAux(&p.To, v)

	case ssa.OpAMD64SETNEF:
		t := v.RegTmp()
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPS)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = t
		// ORL avoids partial register write and is smaller than ORQ, used by old compiler
		opregreg(s, x86.AORL, v.Reg(), t)

	case ssa.OpAMD64SETEQF:
		t := v.RegTmp()
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPC)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = t
		// ANDL avoids partial register write and is smaller than ANDQ, used by old compiler
		opregreg(s, x86.AANDL, v.Reg(), t)

	case ssa.OpAMD64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.OpAMD64FlagEQ, ssa.OpAMD64FlagLT_ULT, ssa.OpAMD64FlagLT_UGT, ssa.OpAMD64FlagGT_ULT, ssa.OpAMD64FlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssa.OpAMD64AddTupleFirst32, ssa.OpAMD64AddTupleFirst64:
		v.Fatalf("AddTupleFirst* should never make it to codegen %v", v.LongString())
	case ssa.OpAMD64REPSTOSQ:
		s.Prog(x86.AREP)
		s.Prog(x86.ASTOSQ)
	case ssa.OpAMD64REPMOVSQ:
		s.Prog(x86.AREP)
		s.Prog(x86.AMOVSQ)
	case ssa.OpAMD64LoweredNilCheck:
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
	case ssa.OpAMD64MOVBatomicload, ssa.OpAMD64MOVLatomicload, ssa.OpAMD64MOVQatomicload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpAMD64XCHGB, ssa.OpAMD64XCHGL, ssa.OpAMD64XCHGQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg0()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpAMD64XADDLlock, ssa.OpAMD64XADDQlock:
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Reg0()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpAMD64CMPXCHGLlock, ssa.OpAMD64CMPXCHGQlock:
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
	case ssa.OpAMD64ANDBlock, ssa.OpAMD64ANDLlock, ssa.OpAMD64ANDQlock, ssa.OpAMD64ORBlock, ssa.OpAMD64ORLlock, ssa.OpAMD64ORQlock:
		// Atomic memory operations that don't need to return the old value.
		s.Prog(x86.ALOCK)
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpAMD64LoweredAtomicAnd64, ssa.OpAMD64LoweredAtomicOr64, ssa.OpAMD64LoweredAtomicAnd32, ssa.OpAMD64LoweredAtomicOr32:
		// Atomic memory operations that need to return the old value.
		// We need to do these with compare-and-exchange to get access to the old value.
		// loop:
		// MOVQ mask, tmp
		// MOVQ (addr), AX
		// ANDQ AX, tmp
		// LOCK CMPXCHGQ tmp, (addr) : note that AX is implicit old value to compare against
		// JNE loop
		// : result in AX
		mov := x86.AMOVQ
		op := x86.AANDQ
		cmpxchg := x86.ACMPXCHGQ
		switch v.Op {
		case ssa.OpAMD64LoweredAtomicOr64:
			op = x86.AORQ
		case ssa.OpAMD64LoweredAtomicAnd32:
			mov = x86.AMOVL
			op = x86.AANDL
			cmpxchg = x86.ACMPXCHGL
		case ssa.OpAMD64LoweredAtomicOr32:
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
	case ssa.OpAMD64PrefetchT0, ssa.OpAMD64PrefetchNTA:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
	case ssa.OpClobber:
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
	case ssa.OpClobberReg:
		x := uint64(0xdeaddeaddeaddead)
		p := s.Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(x)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	// XXX SIMD
	// XXX may change depending on how we handle aliased registers
	case ssa.OpAMD64Zero128, ssa.OpAMD64Zero256, ssa.OpAMD64Zero512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v)
		p.AddRestSourceReg(simdReg(v))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)
	case ssa.OpAMD64VPADDD4:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[0])
		p.AddRestSourceReg(simdReg(v.Args[1]))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)
	case ssa.OpAMD64VMOVDQUload128, ssa.OpAMD64VMOVDQUload256, ssa.OpAMD64VMOVDQUload512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)
	case ssa.OpAMD64VMOVDQUstore128, ssa.OpAMD64VMOVDQUstore256, ssa.OpAMD64VMOVDQUstore512:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)

	case ssa.OpAMD64VPMOVMToVec8x16,
		ssa.OpAMD64VPMOVMToVec8x32,
		ssa.OpAMD64VPMOVMToVec8x64,
		ssa.OpAMD64VPMOVMToVec16x8,
		ssa.OpAMD64VPMOVMToVec16x16,
		ssa.OpAMD64VPMOVMToVec16x32,
		ssa.OpAMD64VPMOVMToVec32x4,
		ssa.OpAMD64VPMOVMToVec32x8,
		ssa.OpAMD64VPMOVMToVec32x16,
		ssa.OpAMD64VPMOVMToVec64x2,
		ssa.OpAMD64VPMOVMToVec64x4,
		ssa.OpAMD64VPMOVMToVec64x8:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = simdReg(v)

	case ssa.OpAMD64VPMOVVec8x16ToM,
		ssa.OpAMD64VPMOVVec8x32ToM,
		ssa.OpAMD64VPMOVVec8x64ToM,
		ssa.OpAMD64VPMOVVec16x8ToM,
		ssa.OpAMD64VPMOVVec16x16ToM,
		ssa.OpAMD64VPMOVVec16x32ToM,
		ssa.OpAMD64VPMOVVec32x4ToM,
		ssa.OpAMD64VPMOVVec32x8ToM,
		ssa.OpAMD64VPMOVVec32x16ToM,
		ssa.OpAMD64VPMOVVec64x2ToM,
		ssa.OpAMD64VPMOVVec64x4ToM,
		ssa.OpAMD64VPMOVVec64x8ToM:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = simdReg(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	default:
		if !ssaGenSIMDValue(s, v) {
			v.Fatalf("genValue not implemented: %s", v.LongString())
		}
	}
}

// Example instruction: VRSQRTPS X1, X1
func simdFp11(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[0])
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPSUBD X1, X2, X3
func simdFp21(s *ssagen.State, v *ssa.Value) *obj.Prog {
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
func simdFpXfp(s *ssagen.State, v *ssa.Value) *obj.Prog {
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
func simdFp2k(s *ssagen.State, v *ssa.Value) *obj.Prog {
	// simdReg handles mask and vector registers altogether
	return simdFp21(s, v)
}

// Example instruction: VPMINUQ X21, X3, K3, X31
func simdFp2kfp(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[1])
	p.AddRestSourceReg(simdReg(v.Args[0]))
	// These "simd*" series of functions assumes:
	// Any "K" register that serves as the write-mask
	// or "predicate" for "predicated AVX512 instructions"
	// sits right at the end of the operand list.
	// TODO: verify this assumption.
	p.AddRestSourceReg(simdReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// This function is to accustomize the shifts.
// The 2nd arg is an XMM, and this function merely checks that.
// Example instruction: VPSLLQ Z1, X1, K1, Z2
func simdFpXkfp(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = v.Args[1].Reg()
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(simdReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPCMPEQW Z26, Z30, K1, K4
func simdFp2kk(s *ssagen.State, v *ssa.Value) *obj.Prog {
	return simdFp2kfp(s, v)
}

// Example instruction: VPOPCNTB X14, K4, X16
func simdFpkfp(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[0])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VROUNDPD $7, X2, X2
func simdFp11Imm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	imm := v.AuxInt
	if imm < 0 || imm > 255 {
		v.Fatalf("Invalid source selection immediate")
	}
	p.From.Offset = imm
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VREDUCEPD $126, X1, K3, X31
func simdFpkfpImm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	imm := v.AuxInt
	if imm < 0 || imm > 255 {
		v.Fatalf("Invalid source selection immediate")
	}
	p.From.Offset = imm
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VCMPPS $7, X2, X9, X2
func simdFp21Imm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	imm := v.AuxInt
	if imm < 0 || imm > 255 {
		v.Fatalf("Invalid source selection immediate")
	}
	p.From.Offset = imm
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPINSRB $3, DX, X0, X0
func simdFpgpfpImm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	imm := v.AuxInt
	if imm < 0 || imm > 255 {
		v.Fatalf("Invalid source selection immediate")
	}
	p.From.Offset = imm
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(v.Args[1].Reg())
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VPCMPD $1, Z1, Z2, K1
func simdFp2kImm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	return simdFp21Imm8(s, v)
}

// Example instruction: VPCMPD $1, Z1, Z2, K2, K1
func simdFp2kkImm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	imm := v.AuxInt
	if imm < 0 || imm > 255 {
		v.Fatalf("Invalid source selection immediate")
	}
	p.From.Offset = imm
	p.From.Type = obj.TYPE_CONST
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(simdReg(v.Args[2]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

func simdFp2kfpImm8(s *ssagen.State, v *ssa.Value) *obj.Prog {
	return simdFp2kkImm8(s, v)
}

// Example instruction: VFMADD213PD Z2, Z1, Z0
func simdFp31ResultInArg0(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Example instruction: VFMADD213PD Z2, Z1, K1, Z0
func simdFp3kfpResultInArg0(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[3]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

// Currently unused
func simdFp31(s *ssagen.State, v *ssa.Value) *obj.Prog {
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
func simdFp3kfp(s *ssagen.State, v *ssa.Value) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdReg(v.Args[2])
	p.AddRestSourceReg(simdReg(v.Args[1]))
	p.AddRestSourceReg(simdReg(v.Args[0]))
	p.AddRestSourceReg(simdReg(v.Args[3]))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdReg(v)
	return p
}

var blockJump = [...]struct {
	asm, invasm obj.As
}{
	ssa.BlockAMD64EQ:  {x86.AJEQ, x86.AJNE},
	ssa.BlockAMD64NE:  {x86.AJNE, x86.AJEQ},
	ssa.BlockAMD64LT:  {x86.AJLT, x86.AJGE},
	ssa.BlockAMD64GE:  {x86.AJGE, x86.AJLT},
	ssa.BlockAMD64LE:  {x86.AJLE, x86.AJGT},
	ssa.BlockAMD64GT:  {x86.AJGT, x86.AJLE},
	ssa.BlockAMD64OS:  {x86.AJOS, x86.AJOC},
	ssa.BlockAMD64OC:  {x86.AJOC, x86.AJOS},
	ssa.BlockAMD64ULT: {x86.AJCS, x86.AJCC},
	ssa.BlockAMD64UGE: {x86.AJCC, x86.AJCS},
	ssa.BlockAMD64UGT: {x86.AJHI, x86.AJLS},
	ssa.BlockAMD64ULE: {x86.AJLS, x86.AJHI},
	ssa.BlockAMD64ORD: {x86.AJPC, x86.AJPS},
	ssa.BlockAMD64NAN: {x86.AJPS, x86.AJPC},
}

var eqfJumps = [2][2]ssagen.IndexJump{
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPS, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPC, Index: 0}}, // next == b.Succs[1]
}
var nefJumps = [2][2]ssagen.IndexJump{
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPC, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPS, Index: 0}}, // next == b.Succs[1]
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	switch b.Kind {
	case ssa.BlockPlain, ssa.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockExit, ssa.BlockRetJmp:
	case ssa.BlockRet:
		s.Prog(obj.ARET)

	case ssa.BlockAMD64EQF:
		s.CombJump(b, next, &eqfJumps)

	case ssa.BlockAMD64NEF:
		s.CombJump(b, next, &nefJumps)

	case ssa.BlockAMD64EQ, ssa.BlockAMD64NE,
		ssa.BlockAMD64LT, ssa.BlockAMD64GE,
		ssa.BlockAMD64LE, ssa.BlockAMD64GT,
		ssa.BlockAMD64OS, ssa.BlockAMD64OC,
		ssa.BlockAMD64ULT, ssa.BlockAMD64UGT,
		ssa.BlockAMD64ULE, ssa.BlockAMD64UGE:
		jmp := blockJump[b.Kind]
		switch next {
		case b.Succs[0].Block():
			s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}

	case ssa.BlockAMD64JUMPTABLE:
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

func loadRegResult(s *ssagen.State, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p := s.Prog(loadByType(t))
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_AUTO
	p.From.Sym = n.Linksym()
	p.From.Offset = n.FrameOffset() + off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = reg
	return p
}

func spillArgReg(pp *objw.Progs, p *obj.Prog, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p = pp.Append(p, storeByType(t), obj.TYPE_REG, reg, 0, obj.TYPE_MEM, 0, n.FrameOffset()+off)
	p.To.Name = obj.NAME_PARAM
	p.To.Sym = n.Linksym()
	p.Pos = p.Pos.WithNotStmt()
	return p
}

// XXX maybe make this part of v.Reg?
// On the other hand, it is architecture-specific.
func simdReg(v *ssa.Value) int16 {
	t := v.Type
	if !t.IsSIMD() {
		panic("simdReg: not a simd type")
	}
	switch t.Size() {
	case 8:
		return v.Reg() // K registers
	case 16:
		return v.Reg()
	case 32:
		return v.Reg() + (x86.REG_Y0 - x86.REG_X0)
	case 64:
		return v.Reg() + (x86.REG_Z0 - x86.REG_X0)
	}
	panic("unreachable")
}

// XXX this is used for shift operations only.
// regalloc will issue OpCopy with incorrect type, but the assigned
// register should be correct, and this function is merely checking
// the sanity of this part.
func simdCheckRegOnly(v *ssa.Value, regStart, regEnd int16) int16 {
	if v.Reg() > regEnd || v.Reg() < regStart {
		panic("simdCheckRegOnly: not the desired register")
	}
	return v.Reg()
}

func simdMov(width int64) obj.As {
	if width >= 64 {
		return x86.AVMOVDQU64
	} else if width >= 16 {
		return x86.AVMOVDQU
	}
	return x86.AKMOVQ
}
