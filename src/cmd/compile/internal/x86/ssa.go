// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"fmt"
	"math"

	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

// markMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *gc.SSAGenState, b *ssa.Block) {
	flive := b.FlagsLiveAtEnd
	if b.Control != nil && b.Control.Type.IsFlags() {
		flive = true
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && v.Op == ssa.Op386MOVLconst {
			// The "mark" is any non-nil Aux value.
			v.Aux = v
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
func loadByType(t ssa.Type) obj.As {
	// Avoid partial register write
	if !t.IsFloat() && t.Size() <= 2 {
		if t.Size() == 1 {
			return x86.AMOVBLZX
		} else {
			return x86.AMOVWLZX
		}
	}
	// Otherwise, there's no difference between load and store opcodes.
	return storeByType(t)
}

// storeByType returns the store instruction of the given type.
func storeByType(t ssa.Type) obj.As {
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
func moveByType(t ssa.Type) obj.As {
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
//     dest := dest(To) op src(From)
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregreg(op obj.As, dest, src int16) *obj.Prog {
	p := gc.Prog(op)
	p.From.Type = obj.TYPE_REG
	p.To.Type = obj.TYPE_REG
	p.To.Reg = dest
	p.From.Reg = src
	return p
}

func ssaGenValue(s *gc.SSAGenState, v *ssa.Value) {
	s.SetLineno(v.Line)

	if gc.Thearch.Use387 {
		if ssaGenValue387(s, v) {
			return // v was handled by 387 generation.
		}
	}

	switch v.Op {
	case ssa.Op386ADDL:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		switch {
		case r == r1:
			p := gc.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r2
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		case r == r2:
			p := gc.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r1
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		default:
			p := gc.Prog(x86.ALEAL)
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
		ssa.Op386ADDSS, ssa.Op386ADDSD, ssa.Op386SUBSS, ssa.Op386SUBSD,
		ssa.Op386MULSS, ssa.Op386MULSD, ssa.Op386DIVSS, ssa.Op386DIVSD,
		ssa.Op386PXOR,
		ssa.Op386ADCL,
		ssa.Op386SBBL:
		r := v.Reg()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		opregreg(v.Op.Asm(), r, v.Args[1].Reg())

	case ssa.Op386ADDLcarry, ssa.Op386SUBLcarry:
		// output 0 is carry/borrow, output 1 is the low 32 bits.
		r := v.Reg0()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output[0] not in same register %s", v.LongString())
		}
		opregreg(v.Op.Asm(), r, v.Args[1].Reg())

	case ssa.Op386ADDLconstcarry, ssa.Op386SUBLconstcarry:
		// output 0 is carry/borrow, output 1 is the low 32 bits.
		r := v.Reg0()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output[0] not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

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

			var c *obj.Prog
			switch v.Op {
			case ssa.Op386DIVL, ssa.Op386MODL:
				c = gc.Prog(x86.ACMPL)
				j = gc.Prog(x86.AJEQ)
				gc.Prog(x86.ACDQ) //TODO: fix

			case ssa.Op386DIVW, ssa.Op386MODW:
				c = gc.Prog(x86.ACMPW)
				j = gc.Prog(x86.AJEQ)
				gc.Prog(x86.ACWD)
			}
			c.From.Type = obj.TYPE_REG
			c.From.Reg = x
			c.To.Type = obj.TYPE_CONST
			c.To.Offset = -1

			j.To.Type = obj.TYPE_BRANCH
		}

		// for unsigned ints, we sign extend by setting DX = 0
		// signed ints were sign extended above
		if v.Op == ssa.Op386DIVLU || v.Op == ssa.Op386MODLU ||
			v.Op == ssa.Op386DIVWU || v.Op == ssa.Op386MODWU {
			c := gc.Prog(x86.AXORL)
			c.From.Type = obj.TYPE_REG
			c.From.Reg = x86.REG_DX
			c.To.Type = obj.TYPE_REG
			c.To.Reg = x86.REG_DX
		}

		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x

		// signed division, rest of the check for -1 case
		if j != nil {
			j2 := gc.Prog(obj.AJMP)
			j2.To.Type = obj.TYPE_BRANCH

			var n *obj.Prog
			if v.Op == ssa.Op386DIVL || v.Op == ssa.Op386DIVW {
				// n * -1 = -n
				n = gc.Prog(x86.ANEGL)
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_AX
			} else {
				// n % -1 == 0
				n = gc.Prog(x86.AXORL)
				n.From.Type = obj.TYPE_REG
				n.From.Reg = x86.REG_DX
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_DX
			}

			j.To.Val = n
			j2.To.Val = s.Pc()
		}

	case ssa.Op386HMULL, ssa.Op386HMULW, ssa.Op386HMULB,
		ssa.Op386HMULLU, ssa.Op386HMULWU, ssa.Op386HMULBU:
		// the frontend rewrites constant division by 8/16/32 bit integers into
		// HMUL by a constant
		// SSA rewrites generate the 64 bit versions

		// Arg[0] is already in AX as it's the only register we allow
		// and DX is the only output we care about (the high bits)
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

		// IMULB puts the high portion in AH instead of DL,
		// so move it to DL for consistency
		if v.Type.Size() == 1 {
			m := gc.Prog(x86.AMOVB)
			m.From.Type = obj.TYPE_REG
			m.From.Reg = x86.REG_AH
			m.To.Type = obj.TYPE_REG
			m.To.Reg = x86.REG_DX
		}

	case ssa.Op386MULLQU:
		// AX * args[1], high 32 bits in DX (result[0]), low 32 bits in AX (result[1]).
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()

	case ssa.Op386ADDLconst:
		r := v.Reg()
		a := v.Args[0].Reg()
		if r == a {
			if v.AuxInt == 1 {
				p := gc.Prog(x86.AINCL)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return
			}
			if v.AuxInt == -1 {
				p := gc.Prog(x86.ADECL)
				p.To.Type = obj.TYPE_REG
				p.To.Reg = r
				return
			}
			p := gc.Prog(v.Op.Asm())
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = v.AuxInt
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			return
		}
		p := gc.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = a
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.Op386MULLconst:
		r := v.Reg()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		// TODO: Teach doasm to compile the three-address multiply imul $c, r1, r2
		// then we don't need to use resultInArg0 for these ops.
		//p.From3 = new(obj.Addr)
		//p.From3.Type = obj.TYPE_REG
		//p.From3.Reg = v.Args[0].Reg()

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
		r := v.Reg()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.Op386SBBLcarrymask:
		r := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.Op386LEAL1, ssa.Op386LEAL2, ssa.Op386LEAL4, ssa.Op386LEAL8:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := gc.Prog(x86.ALEAL)
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
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386LEAL:
		p := gc.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386CMPL, ssa.Op386CMPW, ssa.Op386CMPB,
		ssa.Op386TESTL, ssa.Op386TESTW, ssa.Op386TESTB:
		opregreg(v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssa.Op386UCOMISS, ssa.Op386UCOMISD:
		// Go assembler has swapped operands for UCOMISx relative to CMP,
		// must account for that right here.
		opregreg(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg())
	case ssa.Op386CMPLconst, ssa.Op386CMPWconst, ssa.Op386CMPBconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.Op386TESTLconst, ssa.Op386TESTWconst, ssa.Op386TESTBconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
	case ssa.Op386MOVLconst:
		x := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
		// If flags are live at this instruction, suppress the
		// MOV $0,AX -> XOR AX,AX optimization.
		if v.Aux != nil {
			p.Mark |= x86.PRESERVEFLAGS
		}
	case ssa.Op386MOVSSconst, ssa.Op386MOVSDconst:
		x := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.Op386MOVSSconst1, ssa.Op386MOVSDconst1:
		var literal string
		if v.Op == ssa.Op386MOVSDconst1 {
			literal = fmt.Sprintf("$f64.%016x", uint64(v.AuxInt))
		} else {
			literal = fmt.Sprintf("$f32.%08x", math.Float32bits(float32(math.Float64frombits(uint64(v.AuxInt)))))
		}
		p := gc.Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_EXTERN
		p.From.Sym = obj.Linklookup(gc.Ctxt, literal, 0)
		p.From.Sym.Set(obj.AttrLocal, true)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVSSconst2, ssa.Op386MOVSDconst2:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386MOVSSload, ssa.Op386MOVSDload, ssa.Op386MOVLload, ssa.Op386MOVWload, ssa.Op386MOVBload, ssa.Op386MOVBLSXload, ssa.Op386MOVWLSXload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVSDloadidx8:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.From.Scale = 8
		p.From.Index = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVLloadidx4, ssa.Op386MOVSSloadidx4:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.From.Scale = 4
		p.From.Index = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVWloadidx2:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.From.Scale = 2
		p.From.Index = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVBloadidx1, ssa.Op386MOVWloadidx1, ssa.Op386MOVLloadidx1, ssa.Op386MOVSSloadidx1, ssa.Op386MOVSDloadidx1:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == x86.REG_SP {
			r, i = i, r
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r
		p.From.Scale = 1
		p.From.Index = i
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.Op386MOVSSstore, ssa.Op386MOVSDstore, ssa.Op386MOVLstore, ssa.Op386MOVWstore, ssa.Op386MOVBstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.Op386MOVSDstoreidx8:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = 8
		p.To.Index = v.Args[1].Reg()
		gc.AddAux(&p.To, v)
	case ssa.Op386MOVSSstoreidx4, ssa.Op386MOVLstoreidx4:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = 4
		p.To.Index = v.Args[1].Reg()
		gc.AddAux(&p.To, v)
	case ssa.Op386MOVWstoreidx2:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = 2
		p.To.Index = v.Args[1].Reg()
		gc.AddAux(&p.To, v)
	case ssa.Op386MOVBstoreidx1, ssa.Op386MOVWstoreidx1, ssa.Op386MOVLstoreidx1, ssa.Op386MOVSSstoreidx1, ssa.Op386MOVSDstoreidx1:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == x86.REG_SP {
			r, i = i, r
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r
		p.To.Scale = 1
		p.To.Index = i
		gc.AddAux(&p.To, v)
	case ssa.Op386MOVLstoreconst, ssa.Op386MOVWstoreconst, ssa.Op386MOVBstoreconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux2(&p.To, v, sc.Off())
	case ssa.Op386MOVLstoreconstidx1, ssa.Op386MOVLstoreconstidx4, ssa.Op386MOVWstoreconstidx1, ssa.Op386MOVWstoreconstidx2, ssa.Op386MOVBstoreconstidx1:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val()
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
		case ssa.Op386MOVLstoreconstidx4:
			p.To.Scale = 4
		}
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r
		p.To.Index = i
		gc.AddAux2(&p.To, v, sc.Off())
	case ssa.Op386MOVWLSX, ssa.Op386MOVBLSX, ssa.Op386MOVWLZX, ssa.Op386MOVBLZX,
		ssa.Op386CVTSL2SS, ssa.Op386CVTSL2SD,
		ssa.Op386CVTTSS2SL, ssa.Op386CVTTSD2SL,
		ssa.Op386CVTSS2SD, ssa.Op386CVTSD2SS:
		opregreg(v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssa.Op386DUFFZERO:
		p := gc.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
		p.To.Offset = v.AuxInt
	case ssa.Op386DUFFCOPY:
		p := gc.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))
		p.To.Offset = v.AuxInt

	case ssa.OpCopy, ssa.Op386MOVLconvert: // TODO: use MOVLreg for reg->reg copies instead of OpCopy?
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x != y {
			opregreg(moveByType(v.Type), y, x)
		}
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := gc.Prog(loadByType(v.Type))
		gc.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := gc.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		gc.AddrAuto(&p.To, v)
	case ssa.OpPhi:
		gc.CheckLoweredPhi(v)
	case ssa.OpInitMem:
		// memory arg needs no code
	case ssa.OpArg:
		// input args need no code
	case ssa.Op386LoweredGetClosurePtr:
		// Closure pointer is DX.
		gc.CheckLoweredGetClosurePtr(v)
	case ssa.Op386LoweredGetG:
		r := v.Reg()
		// See the comments in cmd/internal/obj/x86/obj6.go
		// near CanUse1InsnTLS for a detailed explanation of these instructions.
		if x86.CanUse1InsnTLS(gc.Ctxt) {
			// MOVL (TLS), r
			p := gc.Prog(x86.AMOVL)
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		} else {
			// MOVL TLS, r
			// MOVL (r)(TLS*1), r
			p := gc.Prog(x86.AMOVL)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			q := gc.Prog(x86.AMOVL)
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = r
			q.From.Index = x86.REG_TLS
			q.From.Scale = 1
			q.To.Type = obj.TYPE_REG
			q.To.Reg = r
		}
	case ssa.Op386CALLstatic:
		if v.Aux.(*gc.Sym) == gc.Deferreturn.Sym {
			// Deferred calls will appear to be returning to
			// the CALL deferreturn(SB) that we are about to emit.
			// However, the stack trace code will show the line
			// of the instruction byte before the return PC.
			// To avoid that being an unrelated instruction,
			// insert an actual hardware NOP that will have the right line number.
			// This is different from obj.ANOP, which is a virtual no-op
			// that doesn't make it into the instruction stream.
			ginsnop()
		}
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(v.Aux.(*gc.Sym))
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.Op386CALLclosure:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.Op386CALLdefer:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Deferproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.Op386CALLgo:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Newproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.Op386CALLinter:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.Op386NEGL,
		ssa.Op386BSWAPL,
		ssa.Op386NOTL:
		r := v.Reg()
		if r != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.Op386BSFL, ssa.Op386BSFW,
		ssa.Op386BSRL, ssa.Op386BSRW,
		ssa.Op386SQRTSD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpSP, ssa.OpSB, ssa.OpSelect0, ssa.OpSelect1:
		// nothing to do
	case ssa.Op386SETEQ, ssa.Op386SETNE,
		ssa.Op386SETL, ssa.Op386SETLE,
		ssa.Op386SETG, ssa.Op386SETGE,
		ssa.Op386SETGF, ssa.Op386SETGEF,
		ssa.Op386SETB, ssa.Op386SETBE,
		ssa.Op386SETORD, ssa.Op386SETNAN,
		ssa.Op386SETA, ssa.Op386SETAE:
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.Op386SETNEF:
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := gc.Prog(x86.ASETPS)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		opregreg(x86.AORL, v.Reg(), x86.REG_AX)

	case ssa.Op386SETEQF:
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		q := gc.Prog(x86.ASETPC)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		opregreg(x86.AANDL, v.Reg(), x86.REG_AX)

	case ssa.Op386InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.Op386FlagEQ, ssa.Op386FlagLT_ULT, ssa.Op386FlagLT_UGT, ssa.Op386FlagGT_ULT, ssa.Op386FlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssa.Op386REPSTOSL:
		gc.Prog(x86.AREP)
		gc.Prog(x86.ASTOSL)
	case ssa.Op386REPMOVSL:
		gc.Prog(x86.AREP)
		gc.Prog(x86.AMOVSL)
	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
	case ssa.OpKeepAlive:
		gc.KeepAlive(v)
	case ssa.Op386LoweredNilCheck:
		// Issue a load which will fault if the input is nil.
		// TODO: We currently use the 2-byte instruction TESTB AX, (reg).
		// Should we use the 3-byte TESTB $0, (reg) instead?  It is larger
		// but it doesn't have false dependency on AX.
		// Or maybe allocate an output register and use MOVL (reg),reg2 ?
		// That trades clobbering flags for clobbering a register.
		p := gc.Prog(x86.ATESTB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_AX
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
		if gc.Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			gc.Warnl(v.Line, "generated nil check")
		}
	case ssa.Op386FCHS:
		v.Fatalf("FCHS in non-387 mode")
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
	ssa.Block386ULT: {x86.AJCS, x86.AJCC},
	ssa.Block386UGE: {x86.AJCC, x86.AJCS},
	ssa.Block386UGT: {x86.AJHI, x86.AJLS},
	ssa.Block386ULE: {x86.AJLS, x86.AJHI},
	ssa.Block386ORD: {x86.AJPC, x86.AJPS},
	ssa.Block386NAN: {x86.AJPS, x86.AJPC},
}

var eqfJumps = [2][2]gc.FloatingEQNEJump{
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPS, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 1}, {Jump: x86.AJPC, Index: 0}}, // next == b.Succs[1]
}
var nefJumps = [2][2]gc.FloatingEQNEJump{
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPC, Index: 1}}, // next == b.Succs[0]
	{{Jump: x86.AJNE, Index: 0}, {Jump: x86.AJPS, Index: 0}}, // next == b.Succs[1]
}

func ssaGenBlock(s *gc.SSAGenState, b, next *ssa.Block) {
	s.SetLineno(b.Line)

	if gc.Thearch.Use387 {
		// Empty the 387's FP stack before the block ends.
		flush387(s)
	}

	switch b.Kind {
	case ssa.BlockPlain:
		if b.Succs[0].Block() != next {
			p := gc.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockDefer:
		// defer returns in rax:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := gc.Prog(x86.ATESTL)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_AX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x86.REG_AX
		p = gc.Prog(x86.AJNE)
		p.To.Type = obj.TYPE_BRANCH
		s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[1].Block()})
		if b.Succs[0].Block() != next {
			p := gc.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockExit:
		gc.Prog(obj.AUNDEF) // tell plive.go that we never reach here
	case ssa.BlockRet:
		gc.Prog(obj.ARET)
	case ssa.BlockRetJmp:
		p := gc.Prog(obj.AJMP)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(b.Aux.(*gc.Sym))

	case ssa.Block386EQF:
		gc.SSAGenFPJump(s, b, next, &eqfJumps)

	case ssa.Block386NEF:
		gc.SSAGenFPJump(s, b, next, &nefJumps)

	case ssa.Block386EQ, ssa.Block386NE,
		ssa.Block386LT, ssa.Block386GE,
		ssa.Block386LE, ssa.Block386GT,
		ssa.Block386ULT, ssa.Block386UGT,
		ssa.Block386ULE, ssa.Block386UGE:
		jmp := blockJump[b.Kind]
		likely := b.Likely
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = gc.Prog(jmp.invasm)
			likely *= -1
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[1].Block()})
		case b.Succs[1].Block():
			p = gc.Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		default:
			p = gc.Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
			q := gc.Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: q, B: b.Succs[1].Block()})
		}

		// liblink reorders the instruction stream as it sees fit.
		// Pass along what we know so liblink can make use of it.
		// TODO: Once we've fully switched to SSA,
		// make liblink leave our output alone.
		switch likely {
		case ssa.BranchUnlikely:
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 0
		case ssa.BranchLikely:
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
		}

	default:
		b.Fatalf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
