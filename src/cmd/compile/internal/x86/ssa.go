// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"fmt"
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"internal/abi"
)

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {
	flive := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		flive = c.Type.IsFlags() || flive
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && v.Op == ssa.Op386MOVLconst {
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
	} else {
		switch width {
		case 1:
			return x86.AMOVB
		case 2:
			return x86.AMOVW
		case 4:
			return x86.AMOVL
		}
	}
	panic("bad store type")
}

// moveByType returns the reg->reg move instruction of the given type.
func moveByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return x86.AMOVSS
		case 8:
			return x86.AMOVSD
		default:
			panic(fmt.Sprintf("bad float register width %d:%s", t.Size(), t))
		}
	} else {
		switch t.Size() {
		case 1:
			// Avoids partial register write
			return x86.AMOVL
		case 2:
			return x86.AMOVL
		case 4:
			return x86.AMOVL
		default:
			panic(fmt.Sprintf("bad int register width %d:%s", t.Size(), t))
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

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssa.Op386ADDL:
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
			p := s.Prog(x86.ALEAL)
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = r1
			p.From.Scale = 1
			p.From.Index = r2
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}

	// 2-address opcode arithmetic
	case ssa.Op386SUBL,
		ssa.Op386MULL,
		ssa.Op386ANDL,
		ssa.Op386ORL,
		ssa.Op386XORL,
		ssa.Op386SHLL,
		ssa.Op386SHRL, ssa.Op386SHRW, ssa.Op386SHRB,
		ssa.Op386SARL, ssa.Op386SARW, ssa.Op386SARB,
		ssa.Op386ROLL, ssa.Op386ROLW, ssa.Op386ROLB,
		ssa.Op386ADDSS, ssa.Op386ADDSD, ssa.Op386SUBSS, ssa.Op386SUBSD,
		ssa.Op386MULSS, ssa.Op386MULSD, ssa.Op386DIVSS, ssa.Op386DIVSD,
		ssa.Op386PXOR,
		ssa.Op386ADCL,
		ssa.Op386SBBL:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())

	case ssa.Op386ADDLcarry, ssa.Op386SUBLcarry:
		// output 0 is carry/borrow, output 1 is the low 32 bits.
		opregreg(s, v.Op.Asm(), v.Reg0(), v.Args[1].Reg())

	case ssa.Op386ADDLconstcarry, ssa.Op386SUBLconstcarry:
		// output 0 is carry/borrow, output 1 is the low 32 bits.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssa.Op386DIVL, ssa.Op386DIVW,
		ssa.Op386DIVLU, ssa.Op386DIVWU,
		ssa.Op386MODL, ssa.Op386MODW,
		ssa.Op386MODLU, ssa.Op386MODWU:

		// Arg[0] is already in AX as it's the only register we allow
		// and AX is the only output
		x := v.Args[1].Reg()

		// CPU faults upon signed overflow, which occurs when most
		// negative int is divided by -1.
		var j *obj.Prog
		if v.Op == ssa.Op386DIVL || v.Op == ssa.Op386DIVW ||
			v.Op == ssa.Op386MODL || v.Op == ssa.Op386MODW {

			if ssa.DivisionNeedsFixUp(v) {
				var c *obj.Prog
				switch v.Op {
				case ssa.Op386DIVL, ssa.Op386MODL:
					c = s.Prog(x86.ACMPL)
					j = s.Prog(x86.AJEQ)

				case ssa.Op386DIVW, ssa.Op386MODW:
					c = s.Prog(x86.ACMPW)
					j = s.Prog(x86.AJEQ)
				}
				c.From.Type = obj.TYPE_REG
				c.From.Reg = x
				c.To.Type = obj.TYPE_CONST
				c.To.Offset = -1

				j.To.Type = obj.TYPE_BRANCH
			}
			// sign extend the dividend
			switch v.Op {
			case ssa.Op386DIVL, ssa.Op386MODL:
				s.Prog(x86.ACDQ)
			case ssa.Op386DIVW, ssa.Op386MODW:
				s.Prog(x86.ACWD)
			}
		}

		// for unsigned ints, we sign extend by setting DX = 0
		// signed ints were sign extended above
		if v.Op == ssa.Op386DIVLU || v.Op == ssa.Op386MODLU ||
			v.Op == ssa.Op386DIVWU || v.Op == ssa.Op386MODWU {
			c := s.Prog(x86.AXORL)
			c.From.Type = obj.TYPE_REG
			c.From.Reg = x86.REG_DX
			c.To.Type = obj.TYPE_REG
			c.To.Reg = x86.REG_DX
		}

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x

		// signed division, rest of the check for -1 case
		if j != nil {
			j2 := s.Prog(obj.AJMP)
			j2.To.Type = obj.TYPE_BRANCH

			var n *obj.Prog
			if v.Op == ssa.Op386DIVL || v.Op == ssa.Op386DIVW {
				// n * -1 = -n
				n = s.Prog(x86.ANEGL)
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_AX
			} else {
				// n % -1 == 0
				n = s.Prog(x86.AXORL)
				n.From.Type = obj.TYPE_REG
				n.From.Reg = x86.REG_DX
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_DX
			}

			j.To.SetTarget(n)
			j2.To.SetTarget(s.Pc())
		}

	case ssa.Op386HMULL, ssa.Op386HMULLU:
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

	case ssa.Op386MULLU:
		// Arg[0] is already in AX as it's the only register we allow
		// results lo in AX
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssa.Op386MULLQU:
		// AX * args[1], high 32 bits in DX (result[0]), low 32 bits in AX (result[1]).
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssa.Op386AVGLU:
		// compute (x+y)/2 unsigned.
		// Do a 32-bit add, the overflow goes into the carry.
		// Shift right once and pull the carry back into the 31st bit.
		p := s.Prog(x86.AADDL)
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p.From.Reg = v.Args[1].Reg()
		p = s.Prog(x86.ARCRL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386ADDLconst:
		r := v.Reg()
		a := v.Args[0].Reg()
		if r == a {
			if v.AuxInt == 1 {
				p := s.Prog(x86.AINCL)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return
			}
			if v.AuxInt == -1 {
				p := s.Prog(x86.ADECL)
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
		p := s.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = a
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.Op386MULLconst:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.AddRestSourceReg(v.Args[0].Reg())

	case ssa.Op386SUBLconst,
		ssa.Op386ADCLconst,
		ssa.Op386SBBLconst,
		ssa.Op386ANDLconst,
		ssa.Op386ORLconst,
		ssa.Op386XORLconst,
		ssa.Op386SHLLconst,
		ssa.Op386SHRLconst, ssa.Op386SHRWconst, ssa.Op386SHRBconst,
		ssa.Op386SARLconst, ssa.Op386SARWconst, ssa.Op386SARBconst,
		ssa.Op386ROLLconst, ssa.Op386ROLWconst, ssa.Op386ROLBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386SBBLcarrymask:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.Op386LEAL1, ssa.Op386LEAL2, ssa.Op386LEAL4, ssa.Op386LEAL8:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := s.Prog(x86.ALEAL)
		switch v.Op {
		case ssa.Op386LEAL1:
			p.From.Scale = 1
			if i == x86.REG_SP {
				r, i = i, r
			}
		case ssa.Op386LEAL2:
			p.From.Scale = 2
		case ssa.Op386LEAL4:
			p.From.Scale = 4
		case ssa.Op386LEAL8:
			p.From.Scale = 8
		}
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386LEAL:
		p := s.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386CMPL, ssa.Op386CMPW, ssa.Op386CMPB,
		ssa.Op386TESTL, ssa.Op386TESTW, ssa.Op386TESTB:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssa.Op386UCOMISS, ssa.Op386UCOMISD:
		// Go assembler has swapped operands for UCOMISx relative to CMP,
		// must account for that right here.
		opregreg(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg())
	case ssa.Op386CMPLconst, ssa.Op386CMPWconst, ssa.Op386CMPBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.Op386TESTLconst, ssa.Op386TESTWconst, ssa.Op386TESTBconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
	case ssa.Op386CMPLload, ssa.Op386CMPWload, ssa.Op386CMPBload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[1].Reg()
	case ssa.Op386CMPLconstload, ssa.Op386CMPWconstload, ssa.Op386CMPBconstload:
		sc := v.AuxValAndOff()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.From, v, sc.Off64())
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = sc.Val64()
	case ssa.Op386MOVLconst:
		x := v.Reg()

		// If flags aren't live (indicated by v.Aux == nil),
		// then we can rewrite MOV $0, AX into XOR AX, AX.
		if v.AuxInt == 0 && v.Aux == nil {
			p := s.Prog(x86.AXORL)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = x
			break
		}

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.Op386MOVSSconst, ssa.Op386MOVSDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.Op386MOVSSconst1, ssa.Op386MOVSDconst1:
		p := s.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_EXTERN
		f := math.Float64frombits(uint64(v.AuxInt))
		if v.Op == ssa.Op386MOVSDconst1 {
			p.From.Sym = base.Ctxt.Float64Sym(f)
		} else {
			p.From.Sym = base.Ctxt.Float32Sym(float32(f))
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVSSconst2, ssa.Op386MOVSDconst2:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386MOVSSload, ssa.Op386MOVSDload, ssa.Op386MOVLload, ssa.Op386MOVWload, ssa.Op386MOVBload, ssa.Op386MOVBLSXload, ssa.Op386MOVWLSXload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVBloadidx1, ssa.Op386MOVWloadidx1, ssa.Op386MOVLloadidx1, ssa.Op386MOVSSloadidx1, ssa.Op386MOVSDloadidx1,
		ssa.Op386MOVSDloadidx8, ssa.Op386MOVLloadidx4, ssa.Op386MOVSSloadidx4, ssa.Op386MOVWloadidx2:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		switch v.Op {
		case ssa.Op386MOVBloadidx1, ssa.Op386MOVWloadidx1, ssa.Op386MOVLloadidx1, ssa.Op386MOVSSloadidx1, ssa.Op386MOVSDloadidx1:
			if i == x86.REG_SP {
				r, i = i, r
			}
			p.From.Scale = 1
		case ssa.Op386MOVSDloadidx8:
			p.From.Scale = 8
		case ssa.Op386MOVLloadidx4, ssa.Op386MOVSSloadidx4:
			p.From.Scale = 4
		case ssa.Op386MOVWloadidx2:
			p.From.Scale = 2
		}
		p.From.Reg = r
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386ADDLloadidx4, ssa.Op386SUBLloadidx4, ssa.Op386MULLloadidx4,
		ssa.Op386ANDLloadidx4, ssa.Op386ORLloadidx4, ssa.Op386XORLloadidx4:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		p.From.Index = v.Args[2].Reg()
		p.From.Scale = 4
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386ADDLload, ssa.Op386SUBLload, ssa.Op386MULLload,
		ssa.Op386ANDLload, ssa.Op386ORLload, ssa.Op386XORLload,
		ssa.Op386ADDSDload, ssa.Op386ADDSSload, ssa.Op386SUBSDload, ssa.Op386SUBSSload,
		ssa.Op386MULSDload, ssa.Op386MULSSload, ssa.Op386DIVSSload, ssa.Op386DIVSDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVSSstore, ssa.Op386MOVSDstore, ssa.Op386MOVLstore, ssa.Op386MOVWstore, ssa.Op386MOVBstore,
		ssa.Op386ADDLmodify, ssa.Op386SUBLmodify, ssa.Op386ANDLmodify, ssa.Op386ORLmodify, ssa.Op386XORLmodify:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.Op386ADDLconstmodify:
		sc := v.AuxValAndOff()
		val := sc.Val()
		if val == 1 || val == -1 {
			var p *obj.Prog
			if val == 1 {
				p = s.Prog(x86.AINCL)
			} else {
				p = s.Prog(x86.ADECL)
			}
			off := sc.Off64()
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = v.Args[0].Reg()
			ssagen.AddAux2(&p.To, v, off)
			break
		}
		fallthrough
	case ssa.Op386ANDLconstmodify, ssa.Op386ORLconstmodify, ssa.Op386XORLconstmodify:
		sc := v.AuxValAndOff()
		off := sc.Off64()
		val := sc.Val64()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = val
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, off)
	case ssa.Op386MOVBstoreidx1, ssa.Op386MOVWstoreidx1, ssa.Op386MOVLstoreidx1, ssa.Op386MOVSSstoreidx1, ssa.Op386MOVSDstoreidx1,
		ssa.Op386MOVSDstoreidx8, ssa.Op386MOVSSstoreidx4, ssa.Op386MOVLstoreidx4, ssa.Op386MOVWstoreidx2,
		ssa.Op386ADDLmodifyidx4, ssa.Op386SUBLmodifyidx4, ssa.Op386ANDLmodifyidx4, ssa.Op386ORLmodifyidx4, ssa.Op386XORLmodifyidx4:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		switch v.Op {
		case ssa.Op386MOVBstoreidx1, ssa.Op386MOVWstoreidx1, ssa.Op386MOVLstoreidx1, ssa.Op386MOVSSstoreidx1, ssa.Op386MOVSDstoreidx1:
			if i == x86.REG_SP {
				r, i = i, r
			}
			p.To.Scale = 1
		case ssa.Op386MOVSDstoreidx8:
			p.To.Scale = 8
		case ssa.Op386MOVSSstoreidx4, ssa.Op386MOVLstoreidx4,
			ssa.Op386ADDLmodifyidx4, ssa.Op386SUBLmodifyidx4, ssa.Op386ANDLmodifyidx4, ssa.Op386ORLmodifyidx4, ssa.Op386XORLmodifyidx4:
			p.To.Scale = 4
		case ssa.Op386MOVWstoreidx2:
			p.To.Scale = 2
		}
		p.To.Reg = r
		p.To.Index = i
		ssagen.AddAux(&p.To, v)
	case ssa.Op386MOVLstoreconst, ssa.Op386MOVWstoreconst, ssa.Op386MOVBstoreconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.Op386ADDLconstmodifyidx4:
		sc := v.AuxValAndOff()
		val := sc.Val()
		if val == 1 || val == -1 {
			var p *obj.Prog
			if val == 1 {
				p = s.Prog(x86.AINCL)
			} else {
				p = s.Prog(x86.ADECL)
			}
			off := sc.Off64()
			p.To.Type = obj.TYPE_MEM
			p.To.Reg = v.Args[0].Reg()
			p.To.Scale = 4
			p.To.Index = v.Args[1].Reg()
			ssagen.AddAux2(&p.To, v, off)
			break
		}
		fallthrough
	case ssa.Op386MOVLstoreconstidx1, ssa.Op386MOVLstoreconstidx4, ssa.Op386MOVWstoreconstidx1, ssa.Op386MOVWstoreconstidx2, ssa.Op386MOVBstoreconstidx1,
		ssa.Op386ANDLconstmodifyidx4, ssa.Op386ORLconstmodifyidx4, ssa.Op386XORLconstmodifyidx4:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		switch v.Op {
		case ssa.Op386MOVBstoreconstidx1, ssa.Op386MOVWstoreconstidx1, ssa.Op386MOVLstoreconstidx1:
			p.To.Scale = 1
			if i == x86.REG_SP {
				r, i = i, r
			}
		case ssa.Op386MOVWstoreconstidx2:
			p.To.Scale = 2
		case ssa.Op386MOVLstoreconstidx4,
			ssa.Op386ADDLconstmodifyidx4, ssa.Op386ANDLconstmodifyidx4, ssa.Op386ORLconstmodifyidx4, ssa.Op386XORLconstmodifyidx4:
			p.To.Scale = 4
		}
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r
		p.To.Index = i
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.Op386MOVWLSX, ssa.Op386MOVBLSX, ssa.Op386MOVWLZX, ssa.Op386MOVBLZX,
		ssa.Op386CVTSL2SS, ssa.Op386CVTSL2SD,
		ssa.Op386CVTTSS2SL, ssa.Op386CVTTSD2SL,
		ssa.Op386CVTSS2SD, ssa.Op386CVTSD2SS:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssa.Op386DUFFZERO:
		p := s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt
	case ssa.Op386DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt

	case ssa.OpCopy: // TODO: use MOVLreg for reg->reg copies instead of OpCopy?
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
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
		p.To.Reg = v.Reg()

	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddrAuto(&p.To, v)
	case ssa.Op386LoweredGetClosurePtr:
		// Closure pointer is DX.
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssa.Op386LoweredGetG:
		r := v.Reg()
		// See the comments in cmd/internal/obj/x86/obj6.go
		// near CanUse1InsnTLS for a detailed explanation of these instructions.
		if x86.CanUse1InsnTLS(base.Ctxt) {
			// MOVL (TLS), r
			p := s.Prog(x86.AMOVL)
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		} else {
			// MOVL TLS, r
			// MOVL (r)(TLS*1), r
			p := s.Prog(x86.AMOVL)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			q := s.Prog(x86.AMOVL)
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = r
			q.From.Index = x86.REG_TLS
			q.From.Scale = 1
			q.To.Type = obj.TYPE_REG
			q.To.Reg = r
		}

	case ssa.Op386LoweredGetCallerPC:
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = -4 // PC is stored 4 bytes below first parameter.
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386LoweredGetCallerSP:
		// caller's SP is the address of the first arg
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize // 0 on 386, just to be consistent with other architectures
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssa.Op386LoweredPanicBoundsRR, ssa.Op386LoweredPanicBoundsRC, ssa.Op386LoweredPanicBoundsCR, ssa.Op386LoweredPanicBoundsCC,
		ssa.Op386LoweredPanicExtendRR, ssa.Op386LoweredPanicExtendRC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		extend := false
		switch v.Op {
		case ssa.Op386LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - x86.REG_AX)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - x86.REG_AX)
		case ssa.Op386LoweredPanicExtendRR:
			extend = true
			xIsReg = true
			hi := int(v.Args[0].Reg() - x86.REG_AX)
			lo := int(v.Args[1].Reg() - x86.REG_AX)
			xVal = hi<<2 + lo // encode 2 register numbers
			yIsReg = true
			yVal = int(v.Args[2].Reg() - x86.REG_AX)
		case ssa.Op386LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - x86.REG_AX)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(yVal)
			}
		case ssa.Op386LoweredPanicExtendRC:
			extend = true
			xIsReg = true
			hi := int(v.Args[0].Reg() - x86.REG_AX)
			lo := int(v.Args[1].Reg() - x86.REG_AX)
			xVal = hi<<2 + lo // encode 2 register numbers
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				for yVal == hi || yVal == lo {
					yVal++
				}
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(yVal)
			}
		case ssa.Op386LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - x86.REG_AX)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else if signed && int64(int32(c)) == c || !signed && int64(uint32(c)) == c {
				// Move constant to a register
				xIsReg = true
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(xVal)
			} else {
				// Move constant to two registers
				extend = true
				xIsReg = true
				hi := 0
				lo := 1
				if hi == yVal {
					hi = 2
				}
				if lo == yVal {
					lo = 2
				}
				xVal = hi<<2 + lo
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c >> 32
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(hi)
				p = s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(int32(c))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(lo)
			}
		case ssa.Op386LoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else if signed && int64(int32(c)) == c || !signed && int64(uint32(c)) == c {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(xVal)
			} else {
				// Move constant to two registers
				extend = true
				xIsReg = true
				hi := 0
				lo := 1
				xVal = hi<<2 + lo
				p := s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c >> 32
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(hi)
				p = s.Prog(x86.AMOVL)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(int32(c))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = x86.REG_AX + int16(lo)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 2
				p := s.Prog(x86.AMOVL)
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
		if extend {
			p.To.Sym = ir.Syms.PanicExtend
		} else {
			p.To.Sym = ir.Syms.PanicBounds
		}

	case ssa.Op386CALLstatic, ssa.Op386CALLclosure, ssa.Op386CALLinter:
		s.Call(v)
	case ssa.Op386CALLtail:
		s.TailCall(v)
	case ssa.Op386NEGL,
		ssa.Op386BSWAPL,
		ssa.Op386NOTL:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386BSFL, ssa.Op386BSFW,
		ssa.Op386BSRL, ssa.Op386BSRW,
		ssa.Op386SQRTSS, ssa.Op386SQRTSD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386SETEQ, ssa.Op386SETNE,
		ssa.Op386SETL, ssa.Op386SETLE,
		ssa.Op386SETG, ssa.Op386SETGE,
		ssa.Op386SETGF, ssa.Op386SETGEF,
		ssa.Op386SETB, ssa.Op386SETBE,
		ssa.Op386SETORD, ssa.Op386SETNAN,
		ssa.Op386SETA, ssa.Op386SETAE,
		ssa.Op386SETO:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386SETNEF:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPS)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		opregreg(s, x86.AORL, v.Reg(), x86.REG_AX)

	case ssa.Op386SETEQF:
		p := s.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := s.Prog(x86.ASETPC)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		opregreg(s, x86.AANDL, v.Reg(), x86.REG_AX)

	case ssa.Op386InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.Op386FlagEQ, ssa.Op386FlagLT_ULT, ssa.Op386FlagLT_UGT, ssa.Op386FlagGT_ULT, ssa.Op386FlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssa.Op386REPSTOSL:
		s.Prog(x86.AREP)
		s.Prog(x86.ASTOSL)
	case ssa.Op386REPMOVSL:
		s.Prog(x86.AREP)
		s.Prog(x86.AMOVSL)
	case ssa.Op386LoweredNilCheck:
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
		ssagen.AddAux(&p.To, v)
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssa.Op386LoweredCtz32:
		// BSFL in, out
		p := s.Prog(x86.ABSFL)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		// JNZ 2(PC)
		p1 := s.Prog(x86.AJNE)
		p1.To.Type = obj.TYPE_BRANCH

		// MOVL $32, out
		p2 := s.Prog(x86.AMOVL)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = 32
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg()

		// NOP (so the JNZ has somewhere to land)
		nop := s.Prog(obj.ANOP)
		p1.To.SetTarget(nop)
	case ssa.Op386LoweredCtz64:
		if v.Args[0].Reg() == v.Reg() {
			v.Fatalf("input[0] and output in the same register %s", v.LongString())
		}
		if v.Args[1].Reg() == v.Reg() {
			v.Fatalf("input[1] and output in the same register %s", v.LongString())
		}

		// BSFL arg0, out
		p := s.Prog(x86.ABSFL)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		// JNZ 5(PC)
		p1 := s.Prog(x86.AJNE)
		p1.To.Type = obj.TYPE_BRANCH

		// BSFL arg1, out
		p2 := s.Prog(x86.ABSFL)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg()

		// JNZ 2(PC)
		p3 := s.Prog(x86.AJNE)
		p3.To.Type = obj.TYPE_BRANCH

		// MOVL $32, out
		p4 := s.Prog(x86.AMOVL)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = 32
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg()

		// ADDL $32, out
		p5 := s.Prog(x86.AADDL)
		p5.From.Type = obj.TYPE_CONST
		p5.From.Offset = 32
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = v.Reg()
		p3.To.SetTarget(p5)

		// NOP (so the JNZ has somewhere to land)
		nop := s.Prog(obj.ANOP)
		p1.To.SetTarget(nop)

	case ssa.OpClobber:
		p := s.Prog(x86.AMOVL)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0xdeaddead
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = x86.REG_SP
		ssagen.AddAux(&p.To, v)
	case ssa.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = [...]struct {
	asm, invasm obj.As
}{
	ssa.Block386EQ:  {x86.AJEQ, x86.AJNE},
	ssa.Block386NE:  {x86.AJNE, x86.AJEQ},
	ssa.Block386LT:  {x86.AJLT, x86.AJGE},
	ssa.Block386GE:  {x86.AJGE, x86.AJLT},
	ssa.Block386LE:  {x86.AJLE, x86.AJGT},
	ssa.Block386GT:  {x86.AJGT, x86.AJLE},
	ssa.Block386OS:  {x86.AJOS, x86.AJOC},
	ssa.Block386OC:  {x86.AJOC, x86.AJOS},
	ssa.Block386ULT: {x86.AJCS, x86.AJCC},
	ssa.Block386UGE: {x86.AJCC, x86.AJCS},
	ssa.Block386UGT: {x86.AJHI, x86.AJLS},
	ssa.Block386ULE: {x86.AJLS, x86.AJHI},
	ssa.Block386ORD: {x86.AJPC, x86.AJPS},
	ssa.Block386NAN: {x86.AJPS, x86.AJPC},
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

	case ssa.Block386EQF:
		s.CombJump(b, next, &eqfJumps)

	case ssa.Block386NEF:
		s.CombJump(b, next, &nefJumps)

	case ssa.Block386EQ, ssa.Block386NE,
		ssa.Block386LT, ssa.Block386GE,
		ssa.Block386LE, ssa.Block386GT,
		ssa.Block386OS, ssa.Block386OC,
		ssa.Block386ULT, ssa.Block386UGT,
		ssa.Block386ULE, ssa.Block386UGE:
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
	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}
