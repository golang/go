// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
	"math"
)

var ssaRegToReg = []int16{
	// ppc64.REGZERO, // not an SSA reg
	ppc64.REGSP,
	ppc64.REG_R2,
	ppc64.REG_R3,
	ppc64.REG_R4,
	ppc64.REG_R5,
	ppc64.REG_R6,
	ppc64.REG_R7,
	ppc64.REG_R8,
	ppc64.REG_R9,
	ppc64.REG_R10,
	ppc64.REGCTXT,
	ppc64.REG_R12,
	ppc64.REG_R13,
	ppc64.REG_R14,
	ppc64.REG_R15,
	ppc64.REG_R16,
	ppc64.REG_R17,
	ppc64.REG_R18,
	ppc64.REG_R19,
	ppc64.REG_R20,
	ppc64.REG_R21,
	ppc64.REG_R22,
	ppc64.REG_R23,
	ppc64.REG_R24,
	ppc64.REG_R25,
	ppc64.REG_R26,
	ppc64.REG_R27,
	ppc64.REG_R28,
	ppc64.REG_R29,
	ppc64.REGG,
	ppc64.REGTMP,

	ppc64.REG_F0,
	ppc64.REG_F1,
	ppc64.REG_F2,
	ppc64.REG_F3,
	ppc64.REG_F4,
	ppc64.REG_F5,
	ppc64.REG_F6,
	ppc64.REG_F7,
	ppc64.REG_F8,
	ppc64.REG_F9,
	ppc64.REG_F10,
	ppc64.REG_F11,
	ppc64.REG_F12,
	ppc64.REG_F13,
	ppc64.REG_F14,
	ppc64.REG_F15,
	ppc64.REG_F16,
	ppc64.REG_F17,
	ppc64.REG_F18,
	ppc64.REG_F19,
	ppc64.REG_F20,
	ppc64.REG_F21,
	ppc64.REG_F22,
	ppc64.REG_F23,
	ppc64.REG_F24,
	ppc64.REG_F25,
	ppc64.REG_F26,
	// ppc64.REG_F27, // reserved for "floating conversion constant"
	// ppc64.REG_F28, // 0.0
	// ppc64.REG_F29, // 0.5
	// ppc64.REG_F30, // 1.0
	// ppc64.REG_F31, // 2.0

	// ppc64.REG_CR0,
	// ppc64.REG_CR1,
	// ppc64.REG_CR2,
	// ppc64.REG_CR3,
	// ppc64.REG_CR4,
	// ppc64.REG_CR5,
	// ppc64.REG_CR6,
	// ppc64.REG_CR7,

	// ppc64.REG_CR,
	// ppc64.REG_XER,
	// ppc64.REG_LR,
	// ppc64.REG_CTR,
}

var condOps = map[ssa.Op]obj.As{
	ssa.OpPPC64Equal:        ppc64.ABEQ,
	ssa.OpPPC64NotEqual:     ppc64.ABNE,
	ssa.OpPPC64LessThan:     ppc64.ABLT,
	ssa.OpPPC64GreaterEqual: ppc64.ABGE,
	ssa.OpPPC64GreaterThan:  ppc64.ABGT,
	ssa.OpPPC64LessEqual:    ppc64.ABLE,

	ssa.OpPPC64FLessThan:     ppc64.ABLT, // 1 branch for FCMP
	ssa.OpPPC64FGreaterThan:  ppc64.ABGT, // 1 branch for FCMP
	ssa.OpPPC64FLessEqual:    ppc64.ABLT, // 2 branches for FCMP <=, second is BEQ
	ssa.OpPPC64FGreaterEqual: ppc64.ABGT, // 2 branches for FCMP >=, second is BEQ
}

// markMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *gc.SSAGenState, b *ssa.Block) {
	//	flive := b.FlagsLiveAtEnd
	//	if b.Control != nil && b.Control.Type.IsFlags() {
	//		flive = true
	//	}
	//	for i := len(b.Values) - 1; i >= 0; i-- {
	//		v := b.Values[i]
	//		if flive && (v.Op == ssa.OpPPC64MOVWconst || v.Op == ssa.OpPPC64MOVDconst) {
	//			// The "mark" is any non-nil Aux value.
	//			v.Aux = v
	//		}
	//		if v.Type.IsFlags() {
	//			flive = false
	//		}
	//		for _, a := range v.Args {
	//			if a.Type.IsFlags() {
	//				flive = true
	//			}
	//		}
	//	}
}

// loadByType returns the load instruction of the given type.
func loadByType(t ssa.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return ppc64.AFMOVS
		case 8:
			return ppc64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return ppc64.AMOVB
			} else {
				return ppc64.AMOVBZ
			}
		case 2:
			if t.IsSigned() {
				return ppc64.AMOVH
			} else {
				return ppc64.AMOVHZ
			}
		case 4:
			if t.IsSigned() {
				return ppc64.AMOVW
			} else {
				return ppc64.AMOVWZ
			}
		case 8:
			return ppc64.AMOVD
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t ssa.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return ppc64.AFMOVS
		case 8:
			return ppc64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return ppc64.AMOVB
		case 2:
			return ppc64.AMOVH
		case 4:
			return ppc64.AMOVW
		case 8:
			return ppc64.AMOVD
		}
	}
	panic("bad store type")
}

// scratchFpMem initializes an Addr (field of a Prog)
// to reference the scratchpad memory for movement between
// F and G registers for FP conversions.
func scratchFpMem(s *gc.SSAGenState, a *obj.Addr) {
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_AUTO
	a.Node = s.ScratchFpMem
	a.Sym = gc.Linksym(s.ScratchFpMem.Sym)
	a.Reg = ppc64.REGSP
}

func ssaGenValue(s *gc.SSAGenState, v *ssa.Value) {
	s.SetLineno(v.Line)
	switch v.Op {
	case ssa.OpInitMem:
		// memory arg needs no code
	case ssa.OpArg:
		// input args need no code
	case ssa.OpSP, ssa.OpSB, ssa.OpGetG:
		// nothing to do

	case ssa.OpCopy, ssa.OpPPC64MOVDconvert:
		t := v.Type
		if t.IsMemory() {
			return
		}
		x := gc.SSARegNum(v.Args[0])
		y := gc.SSARegNum(v)
		if x != y {
			rt := obj.TYPE_REG
			op := ppc64.AMOVD

			if t.IsFloat() {
				op = ppc64.AFMOVD
			}
			p := gc.Prog(op)
			p.From.Type = rt
			p.From.Reg = x
			p.To.Type = rt
			p.To.Reg = y
		}

	case ssa.OpPPC64Xf2i64:
		{
			x := gc.SSARegNum(v.Args[0])
			y := gc.SSARegNum(v)
			p := gc.Prog(ppc64.AFMOVD)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			scratchFpMem(s, &p.To)
			p = gc.Prog(ppc64.AMOVD)
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
			scratchFpMem(s, &p.From)
		}
	case ssa.OpPPC64Xi2f64:
		{
			x := gc.SSARegNum(v.Args[0])
			y := gc.SSARegNum(v)
			p := gc.Prog(ppc64.AMOVD)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			scratchFpMem(s, &p.To)
			p = gc.Prog(ppc64.AFMOVD)
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
			scratchFpMem(s, &p.From)
		}

	case ssa.OpPPC64LoweredGetClosurePtr:
		// Closure pointer is R11 (already)
		gc.CheckLoweredGetClosurePtr(v)

	case ssa.OpLoadReg:
		loadOp := loadByType(v.Type)
		n, off := gc.AutoVar(v.Args[0])
		p := gc.Prog(loadOp)
		p.From.Type = obj.TYPE_MEM
		p.From.Node = n
		p.From.Sym = gc.Linksym(n.Sym)
		p.From.Offset = off
		if n.Class == gc.PPARAM || n.Class == gc.PPARAMOUT {
			p.From.Name = obj.NAME_PARAM
			p.From.Offset += n.Xoffset
		} else {
			p.From.Name = obj.NAME_AUTO
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpStoreReg:
		storeOp := storeByType(v.Type)
		n, off := gc.AutoVar(v)
		p := gc.Prog(storeOp)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.To.Type = obj.TYPE_MEM
		p.To.Node = n
		p.To.Sym = gc.Linksym(n.Sym)
		p.To.Offset = off
		if n.Class == gc.PPARAM || n.Class == gc.PPARAMOUT {
			p.To.Name = obj.NAME_PARAM
			p.To.Offset += n.Xoffset
		} else {
			p.To.Name = obj.NAME_AUTO
		}

	case ssa.OpPPC64DIVD:
		// For now,
		//
		// cmp arg1, -1
		// be  ahead
		// v = arg0 / arg1
		// b over
		// ahead: v = - arg0
		// over: nop
		r := gc.SSARegNum(v)
		r0 := gc.SSARegNum(v.Args[0])
		r1 := gc.SSARegNum(v.Args[1])

		p := gc.Prog(ppc64.ACMP)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = -1

		pbahead := gc.Prog(ppc64.ABEQ)
		pbahead.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

		pbover := gc.Prog(obj.AJMP)
		pbover.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(ppc64.ANEG)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r0
		gc.Patch(pbahead, p)

		p = gc.Prog(obj.ANOP)
		gc.Patch(pbover, p)

	case ssa.OpPPC64DIVW:
		// word-width version of above
		r := gc.SSARegNum(v)
		r0 := gc.SSARegNum(v.Args[0])
		r1 := gc.SSARegNum(v.Args[1])

		p := gc.Prog(ppc64.ACMPW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = -1

		pbahead := gc.Prog(ppc64.ABEQ)
		pbahead.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

		pbover := gc.Prog(obj.AJMP)
		pbover.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(ppc64.ANEG)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r0
		gc.Patch(pbahead, p)

		p = gc.Prog(obj.ANOP)
		gc.Patch(pbover, p)

	case ssa.OpPPC64ADD, ssa.OpPPC64FADD, ssa.OpPPC64FADDS, ssa.OpPPC64SUB, ssa.OpPPC64FSUB, ssa.OpPPC64FSUBS,
		ssa.OpPPC64MULLD, ssa.OpPPC64MULLW, ssa.OpPPC64DIVDU, ssa.OpPPC64DIVWU,
		ssa.OpPPC64SRAD, ssa.OpPPC64SRAW, ssa.OpPPC64SRD, ssa.OpPPC64SRW, ssa.OpPPC64SLD, ssa.OpPPC64SLW,
		ssa.OpPPC64MULHD, ssa.OpPPC64MULHW, ssa.OpPPC64MULHDU, ssa.OpPPC64MULHWU,
		ssa.OpPPC64FMUL, ssa.OpPPC64FMULS, ssa.OpPPC64FDIV, ssa.OpPPC64FDIVS,
		ssa.OpPPC64AND, ssa.OpPPC64OR, ssa.OpPPC64ANDN, ssa.OpPPC64ORN, ssa.OpPPC64XOR, ssa.OpPPC64EQV:
		r := gc.SSARegNum(v)
		r1 := gc.SSARegNum(v.Args[0])
		r2 := gc.SSARegNum(v.Args[1])
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.OpPPC64MaskIfNotCarry:
		r := gc.SSARegNum(v)
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.OpPPC64ADDconstForCarry:
		r1 := gc.SSARegNum(v.Args[0])
		p := gc.Prog(v.Op.Asm())
		p.Reg = r1
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP // Ignored; this is for the carry effect.

	case ssa.OpPPC64NEG, ssa.OpPPC64FNEG, ssa.OpPPC64FSQRT, ssa.OpPPC64FSQRTS, ssa.OpPPC64FCTIDZ, ssa.OpPPC64FCTIWZ, ssa.OpPPC64FCFID, ssa.OpPPC64FRSP:
		r := gc.SSARegNum(v)
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])

	case ssa.OpPPC64ADDconst, ssa.OpPPC64ANDconst, ssa.OpPPC64ORconst, ssa.OpPPC64XORconst,
		ssa.OpPPC64SRADconst, ssa.OpPPC64SRAWconst, ssa.OpPPC64SRDconst, ssa.OpPPC64SRWconst, ssa.OpPPC64SLDconst, ssa.OpPPC64SLWconst:
		p := gc.Prog(v.Op.Asm())
		p.Reg = gc.SSARegNum(v.Args[0])

		if v.Aux != nil {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = gc.AuxOffset(v)
		} else {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = v.AuxInt
		}

		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpPPC64MOVDaddr:
		p := gc.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

		var wantreg string
		// Suspect comment, copied from ARM code
		// MOVD $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP
		//               when constant is large, tmp register (R11) may be used
		// - base is SB: load external address from constant pool (use relocation)
		switch v.Aux.(type) {
		default:
			v.Fatalf("aux is of unknown type %T", v.Aux)
		case *ssa.ExternSymbol:
			wantreg = "SB"
			gc.AddAux(&p.From, v)
		case *ssa.ArgSymbol, *ssa.AutoSymbol:
			wantreg = "SP"
			gc.AddAux(&p.From, v)
		case nil:
			// No sym, just MOVD $off(SP), R
			wantreg = "SP"
			p.From.Reg = ppc64.REGSP
			p.From.Offset = v.AuxInt
		}
		if reg := gc.SSAReg(v.Args[0]); reg.Name() != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg.Name(), v.Aux, wantreg)
		}

	case ssa.OpPPC64MOVDconst, ssa.OpPPC64MOVWconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpPPC64FMOVDconst, ssa.OpPPC64FMOVSconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpPPC64FCMPU, ssa.OpPPC64CMP, ssa.OpPPC64CMPW, ssa.OpPPC64CMPU, ssa.OpPPC64CMPWU:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v.Args[1])

	case ssa.OpPPC64CMPconst, ssa.OpPPC64CMPUconst, ssa.OpPPC64CMPWconst, ssa.OpPPC64CMPWUconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt

	case ssa.OpPPC64MOVBreg, ssa.OpPPC64MOVBZreg, ssa.OpPPC64MOVHreg, ssa.OpPPC64MOVHZreg, ssa.OpPPC64MOVWreg, ssa.OpPPC64MOVWZreg:
		// Shift in register to required size
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.To.Reg = gc.SSARegNum(v)
		p.To.Type = obj.TYPE_REG

	case ssa.OpPPC64MOVDload, ssa.OpPPC64MOVWload, ssa.OpPPC64MOVBload, ssa.OpPPC64MOVHload, ssa.OpPPC64MOVWZload, ssa.OpPPC64MOVBZload, ssa.OpPPC64MOVHZload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpPPC64FMOVDload, ssa.OpPPC64FMOVSload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

	case ssa.OpPPC64MOVDstorezero, ssa.OpPPC64MOVWstorezero, ssa.OpPPC64MOVHstorezero, ssa.OpPPC64MOVBstorezero:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.To, v)

	case ssa.OpPPC64MOVDstore, ssa.OpPPC64MOVWstore, ssa.OpPPC64MOVHstore, ssa.OpPPC64MOVBstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.To, v)
	case ssa.OpPPC64FMOVDstore, ssa.OpPPC64FMOVSstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.To, v)

	case ssa.OpPPC64Equal,
		ssa.OpPPC64NotEqual,
		ssa.OpPPC64LessThan,
		ssa.OpPPC64FLessThan,
		ssa.OpPPC64LessEqual,
		ssa.OpPPC64GreaterThan,
		ssa.OpPPC64FGreaterThan,
		ssa.OpPPC64GreaterEqual:
		// On Power7 or later, can use isel instruction:
		// for a < b, a > b, a = b:
		//   rt := 1
		//   isel rt,rt,r0,cond

		// for  a >= b, a <= b, a != b:
		//   rt := 1
		//   isel rt,0,rt,!cond

		// However, PPCbe support is for older machines than that,
		// and isel (which looks a lot like fsel) isn't recognized
		// yet by the Go assembler.  So for now, use the old instruction
		// sequence, which we'll need anyway.
		// TODO: add support for isel on PPCle and use it.

		// generate boolean values
		// use conditional move

		p := gc.Prog(ppc64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

		pb := gc.Prog(condOps[v.Op])
		pb.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(ppc64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

		p = gc.Prog(obj.ANOP)
		gc.Patch(pb, p)

	case ssa.OpPPC64FLessEqual, // These include a second branch for EQ -- dealing with NaN prevents REL= to !REL conversion
		ssa.OpPPC64FGreaterEqual:

		p := gc.Prog(ppc64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

		pb0 := gc.Prog(condOps[v.Op])
		pb0.To.Type = obj.TYPE_BRANCH
		pb1 := gc.Prog(ppc64.ABEQ)
		pb1.To.Type = obj.TYPE_BRANCH

		p = gc.Prog(ppc64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)

		p = gc.Prog(obj.ANOP)
		gc.Patch(pb0, p)
		gc.Patch(pb1, p)

	case ssa.OpPPC64LoweredZero:
		// Similar to how this is done on ARM,
		// except that PPC MOVDU x,off(y) is *(y+off) = x; y=y+off
		// not store-and-increment.
		// Therefore R3 should be dest-align
		// and arg1 should be dest+size-align
		// HOWEVER, the input dest address cannot be dest-align because
		// that does not necessarily address valid memory and it's not
		// known how that might be optimized.  Therefore, correct it in
		// in the expansion:
		//
		// ADD    -8,R3,R3
		// MOVDU  R0, 8(R3)
		// CMP	  R3, Rarg1
		// BL	  -2(PC)
		// arg1 is the address of the last element to zero
		// auxint is alignment
		var sz int64
		var movu obj.As
		switch {
		case v.AuxInt%8 == 0:
			sz = 8
			movu = ppc64.AMOVDU
		case v.AuxInt%4 == 0:
			sz = 4
			movu = ppc64.AMOVWZU // MOVWU instruction not implemented
		case v.AuxInt%2 == 0:
			sz = 2
			movu = ppc64.AMOVHU
		default:
			sz = 1
			movu = ppc64.AMOVBU
		}

		p := gc.Prog(ppc64.AADD)
		p.Reg = gc.SSARegNum(v.Args[0])
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v.Args[0])

		p = gc.Prog(movu)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REG_R0
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		p.To.Offset = sz

		p2 := gc.Prog(ppc64.ACMPU)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = gc.SSARegNum(v.Args[0])
		p2.To.Reg = gc.SSARegNum(v.Args[1])
		p2.To.Type = obj.TYPE_REG

		p3 := gc.Prog(ppc64.ABLT)
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)

	case ssa.OpPPC64LoweredMove:
		// Similar to how this is done on ARM,
		// except that PPC MOVDU x,off(y) is *(y+off) = x; y=y+off,
		// not store-and-increment.
		// Inputs must be valid pointers to memory,
		// so adjust arg0 and arg1 as part of the expansion.
		// arg2 should be src+size-align,
		//
		// ADD    -8,R3,R3
		// ADD    -8,R4,R4
		// MOVDU	8(R4), Rtmp
		// MOVDU 	Rtmp, 8(R3)
		// CMP	R4, Rarg2
		// BL	-3(PC)
		// arg2 is the address of the last element of src
		// auxint is alignment
		var sz int64
		var movu obj.As
		switch {
		case v.AuxInt%8 == 0:
			sz = 8
			movu = ppc64.AMOVDU
		case v.AuxInt%4 == 0:
			sz = 4
			movu = ppc64.AMOVWZU // MOVWU instruction not implemented
		case v.AuxInt%2 == 0:
			sz = 2
			movu = ppc64.AMOVHU
		default:
			sz = 1
			movu = ppc64.AMOVBU
		}

		p := gc.Prog(ppc64.AADD)
		p.Reg = gc.SSARegNum(v.Args[0])
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v.Args[0])

		p = gc.Prog(ppc64.AADD)
		p.Reg = gc.SSARegNum(v.Args[1])
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v.Args[1])

		p = gc.Prog(movu)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = gc.SSARegNum(v.Args[1])
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP

		p2 := gc.Prog(movu)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = ppc64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = gc.SSARegNum(v.Args[0])
		p2.To.Offset = sz

		p3 := gc.Prog(ppc64.ACMPU)
		p3.From.Reg = gc.SSARegNum(v.Args[1])
		p3.From.Type = obj.TYPE_REG
		p3.To.Reg = gc.SSARegNum(v.Args[2])
		p3.To.Type = obj.TYPE_REG

		p4 := gc.Prog(ppc64.ABLT)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)

	case ssa.OpPPC64CALLstatic:
		if v.Aux.(*gc.Sym) == gc.Deferreturn.Sym {
			// Deferred calls will appear to be returning to
			// the CALL deferreturn(SB) that we are about to emit.
			// However, the stack trace code will show the line
			// of the instruction byte before the return PC.
			// To avoid that being an unrelated instruction,
			// insert two actual hardware NOPs that will have the right line number.
			// This is different from obj.ANOP, which is a virtual no-op
			// that doesn't make it into the instruction stream.
			// PPC64 is unusual because TWO nops are required
			// (see gc/cgen.go, gc/plive.go)
			ginsnop()
			ginsnop()
		}
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(v.Aux.(*gc.Sym))
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpPPC64CALLclosure:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpPPC64CALLdefer:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Deferproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpPPC64CALLgo:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Newproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpPPC64CALLinter:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}

	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
	case ssa.OpKeepAlive:
		if !v.Args[0].Type.IsPtrShaped() {
			v.Fatalf("keeping non-pointer alive %v", v.Args[0])
		}
		n, off := gc.AutoVar(v.Args[0])
		if n == nil {
			v.Fatalf("KeepLive with non-spilled value %s %s", v, v.Args[0])
		}
		if off != 0 {
			v.Fatalf("KeepLive with non-zero offset spill location %s:%d", n, off)
		}
		gc.Gvarlive(n)

	case ssa.OpPhi:
		// just check to make sure regalloc and stackalloc did it right
		if v.Type.IsMemory() {
			return
		}
		f := v.Block.Func
		loc := f.RegAlloc[v.ID]
		for _, a := range v.Args {
			if aloc := f.RegAlloc[a.ID]; aloc != loc { // TODO: .Equal() instead?
				v.Fatalf("phi arg at different location than phi: %v @ %v, but arg %v @ %v\n%s\n", v, loc, a, aloc, v.Block.Func)
			}
		}

	case ssa.OpPPC64LoweredNilCheck:
		// Optimization - if the subsequent block has a load or store
		// at the same address, we don't need to issue this instruction.
		// mem := v.Args[1]
		// for _, w := range v.Block.Succs[0].Block().Values {
		// 	if w.Op == ssa.OpPhi {
		// 		if w.Type.IsMemory() {
		// 			mem = w
		// 		}
		// 		continue
		// 	}
		// 	if len(w.Args) == 0 || !w.Args[len(w.Args)-1].Type.IsMemory() {
		// 		// w doesn't use a store - can't be a memory op.
		// 		continue
		// 	}
		// 	if w.Args[len(w.Args)-1] != mem {
		// 		v.Fatalf("wrong store after nilcheck v=%s w=%s", v, w)
		// 	}
		// 	switch w.Op {
		// 	case ssa.OpPPC64MOVBload, ssa.OpPPC64MOVBUload, ssa.OpPPC64MOVHload, ssa.OpPPC64MOVHUload,
		// 		ssa.OpPPC64MOVWload, ssa.OpPPC64MOVFload, ssa.OpPPC64MOVDload,
		// 		ssa.OpPPC64MOVBstore, ssa.OpPPC64MOVHstore, ssa.OpPPC64MOVWstore,
		// 		ssa.OpPPC64MOVFstore, ssa.OpPPC64MOVDstore:
		// 		// arg0 is ptr, auxint is offset
		// 		if w.Args[0] == v.Args[0] && w.Aux == nil && w.AuxInt >= 0 && w.AuxInt < minZeroPage {
		// 			if gc.Debug_checknil != 0 && int(v.Line) > 1 {
		// 				gc.Warnl(v.Line, "removed nil check")
		// 			}
		// 			return
		// 		}
		// 	case ssa.OpPPC64DUFFZERO, ssa.OpPPC64LoweredZero, ssa.OpPPC64LoweredZeroU:
		// 		// arg0 is ptr
		// 		if w.Args[0] == v.Args[0] {
		// 			if gc.Debug_checknil != 0 && int(v.Line) > 1 {
		// 				gc.Warnl(v.Line, "removed nil check")
		// 			}
		// 			return
		// 		}
		// 	case ssa.OpPPC64DUFFCOPY, ssa.OpPPC64LoweredMove, ssa.OpPPC64LoweredMoveU:
		// 		// arg0 is dst ptr, arg1 is src ptr
		// 		if w.Args[0] == v.Args[0] || w.Args[1] == v.Args[0] {
		// 			if gc.Debug_checknil != 0 && int(v.Line) > 1 {
		// 				gc.Warnl(v.Line, "removed nil check")
		// 			}
		// 			return
		// 		}
		// 	default:
		// 	}
		// 	if w.Type.IsMemory() {
		// 		if w.Op == ssa.OpVarDef || w.Op == ssa.OpVarKill || w.Op == ssa.OpVarLive {
		// 			// these ops are OK
		// 			mem = w
		// 			continue
		// 		}
		// 		// We can't delay the nil check past the next store.
		// 		break
		// 	}
		// }
		// Issue a load which will fault if arg is nil.
		p := gc.Prog(ppc64.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP
		if gc.Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			gc.Warnl(v.Line, "generated nil check")
		}

	case ssa.OpPPC64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.OpPPC64FlagEQ, ssa.OpPPC64FlagLT, ssa.OpPPC64FlagGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())

	default:
		v.Unimplementedf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = [...]struct {
	asm, invasm     obj.As
	asmeq, invasmeq bool
}{
	ssa.BlockPPC64EQ: {ppc64.ABEQ, ppc64.ABNE, false, false},
	ssa.BlockPPC64NE: {ppc64.ABNE, ppc64.ABEQ, false, false},

	ssa.BlockPPC64LT: {ppc64.ABLT, ppc64.ABGE, false, false},
	ssa.BlockPPC64GE: {ppc64.ABGE, ppc64.ABLT, false, false},
	ssa.BlockPPC64LE: {ppc64.ABLE, ppc64.ABGT, false, false},
	ssa.BlockPPC64GT: {ppc64.ABGT, ppc64.ABLE, false, false},

	// TODO: need to work FP comparisons into block jumps
	ssa.BlockPPC64FLT: {ppc64.ABLT, ppc64.ABGT, false, true},
	ssa.BlockPPC64FGE: {ppc64.ABGT, ppc64.ABLT, true, false},
	ssa.BlockPPC64FLE: {ppc64.ABLT, ppc64.ABGT, true, false},
	ssa.BlockPPC64FGT: {ppc64.ABGT, ppc64.ABLT, false, true},
}

func ssaGenBlock(s *gc.SSAGenState, b, next *ssa.Block) {
	s.SetLineno(b.Line)

	switch b.Kind {

	case ssa.BlockDefer:
		// defer returns in R3:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := gc.Prog(ppc64.ACMP)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REG_R3
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REG_R0

		p = gc.Prog(ppc64.ABNE)
		p.To.Type = obj.TYPE_BRANCH
		s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[1].Block()})
		if b.Succs[0].Block() != next {
			p := gc.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		}

	case ssa.BlockPlain, ssa.BlockCall, ssa.BlockCheck:
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

	case ssa.BlockPPC64EQ, ssa.BlockPPC64NE,
		ssa.BlockPPC64LT, ssa.BlockPPC64GE,
		ssa.BlockPPC64LE, ssa.BlockPPC64GT,
		ssa.BlockPPC64FLT, ssa.BlockPPC64FGE,
		ssa.BlockPPC64FLE, ssa.BlockPPC64FGT:
		jmp := blockJump[b.Kind]
		likely := b.Likely
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = gc.Prog(jmp.invasm)
			likely *= -1
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[1].Block()})
			if jmp.invasmeq {
				// TODO: The second branch is probably predict-not-taken since it is for FP equality
				q := gc.Prog(ppc64.ABEQ)
				q.To.Type = obj.TYPE_BRANCH
				s.Branches = append(s.Branches, gc.Branch{P: q, B: b.Succs[1].Block()})
			}
		case b.Succs[1].Block():
			p = gc.Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
			if jmp.asmeq {
				q := gc.Prog(ppc64.ABEQ)
				q.To.Type = obj.TYPE_BRANCH
				s.Branches = append(s.Branches, gc.Branch{P: q, B: b.Succs[0].Block()})
			}
		default:
			p = gc.Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
			if jmp.asmeq {
				q := gc.Prog(ppc64.ABEQ)
				q.To.Type = obj.TYPE_BRANCH
				s.Branches = append(s.Branches, gc.Branch{P: q, B: b.Succs[0].Block()})
			}
			q := gc.Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: q, B: b.Succs[1].Block()})
		}

		// liblink reorders the instruction stream as it sees fit.
		// Pass along what we know so liblink can make use of it.
		// TODO: Once we've fully switched to SSA,
		// make liblink leave our output alone.
		//switch likely {
		//case ssa.BranchUnlikely:
		//	p.From.Type = obj.TYPE_CONST
		//	p.From.Offset = 0
		//case ssa.BranchLikely:
		//	p.From.Type = obj.TYPE_CONST
		//	p.From.Offset = 1
		//}

	default:
		b.Unimplementedf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
