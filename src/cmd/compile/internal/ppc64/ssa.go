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

// iselOp encodes mapping of comparison operations onto ISEL operands
type iselOp struct {
	cond        int64
	valueIfCond int // if cond is true, the value to return (0 or 1)
}

// Input registers to ISEL used for comparison. Index 0 is zero, 1 is (will be) 1
var iselRegs = [2]int16{ppc64.REG_R0, ppc64.REGTMP}

var iselOps = map[ssa.Op]iselOp{
	ssa.OpPPC64Equal:         iselOp{cond: ppc64.C_COND_EQ, valueIfCond: 1},
	ssa.OpPPC64NotEqual:      iselOp{cond: ppc64.C_COND_EQ, valueIfCond: 0},
	ssa.OpPPC64LessThan:      iselOp{cond: ppc64.C_COND_LT, valueIfCond: 1},
	ssa.OpPPC64GreaterEqual:  iselOp{cond: ppc64.C_COND_LT, valueIfCond: 0},
	ssa.OpPPC64GreaterThan:   iselOp{cond: ppc64.C_COND_GT, valueIfCond: 1},
	ssa.OpPPC64LessEqual:     iselOp{cond: ppc64.C_COND_GT, valueIfCond: 0},
	ssa.OpPPC64FLessThan:     iselOp{cond: ppc64.C_COND_LT, valueIfCond: 1},
	ssa.OpPPC64FGreaterThan:  iselOp{cond: ppc64.C_COND_GT, valueIfCond: 1},
	ssa.OpPPC64FLessEqual:    iselOp{cond: ppc64.C_COND_LT, valueIfCond: 1}, // 2 comparisons, 2nd is EQ
	ssa.OpPPC64FGreaterEqual: iselOp{cond: ppc64.C_COND_GT, valueIfCond: 1}, // 2 comparisons, 2nd is EQ
}

// markMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *gc.SSAGenState, b *ssa.Block) {
	//	flive := b.FlagsLiveAtEnd
	//	if b.Control != nil && b.Control.Type.IsFlags() {
	//		flive = true
	//	}
	//	for i := len(b.Values) - 1; i >= 0; i-- {
	//		v := b.Values[i]
	//		if flive && (v.Op == v.Op == ssa.OpPPC64MOVDconst) {
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

func ssaGenISEL(v *ssa.Value, cr int64, r1, r2 int16) {
	r := v.Reg()
	p := gc.Prog(ppc64.AISEL)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = r
	p.Reg = r1
	p.From3 = &obj.Addr{Type: obj.TYPE_REG, Reg: r2}
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = cr
}

func ssaGenValue(s *gc.SSAGenState, v *ssa.Value) {
	switch v.Op {
	case ssa.OpCopy, ssa.OpPPC64MOVDconvert:
		t := v.Type
		if t.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
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
			x := v.Args[0].Reg()
			y := v.Reg()

			p := gc.Prog(ppc64.AMFVSRD)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
		}
	case ssa.OpPPC64Xi2f64:
		{
			x := v.Args[0].Reg()
			y := v.Reg()

			p := gc.Prog(ppc64.AMTVSRD)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
		}

	case ssa.OpPPC64LoweredAtomicAnd8,
		ssa.OpPPC64LoweredAtomicOr8:
		// SYNC
		// LBAR		(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// STBCCC	Rtmp, (Rarg0)
		// BNE		-3(PC)
		// ISYNC
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		p := gc.Prog(ppc64.ALBAR)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP
		p1 := gc.Prog(v.Op.Asm())
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = ppc64.REGTMP
		p2 := gc.Prog(ppc64.ASTBCCC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = ppc64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = r0
		p2.RegTo2 = ppc64.REGTMP
		p3 := gc.Prog(ppc64.ABNE)
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)
		pisync := gc.Prog(ppc64.AISYNC)
		pisync.To.Type = obj.TYPE_NONE

	case ssa.OpPPC64LoweredAtomicAdd32,
		ssa.OpPPC64LoweredAtomicAdd64:
		// SYNC
		// LDAR/LWAR    (Rarg0), Rout
		// ADD		Rarg1, Rout
		// STDCCC/STWCCC Rout, (Rarg0)
		// BNE         -3(PC)
		// ISYNC
		// MOVW		Rout,Rout (if Add32)
		ld := ppc64.ALDAR
		st := ppc64.ASTDCCC
		if v.Op == ssa.OpPPC64LoweredAtomicAdd32 {
			ld = ppc64.ALWAR
			st = ppc64.ASTWCCC
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		// SYNC
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		// LDAR or LWAR
		p := gc.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		// ADD reg1,out
		p1 := gc.Prog(ppc64.AADD)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Reg = out
		p1.To.Type = obj.TYPE_REG
		// STDCCC or STWCCC
		p3 := gc.Prog(st)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = out
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = r0
		// BNE retry
		p4 := gc.Prog(ppc64.ABNE)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)
		// ISYNC
		pisync := gc.Prog(ppc64.AISYNC)
		pisync.To.Type = obj.TYPE_NONE

		// Ensure a 32 bit result
		if v.Op == ssa.OpPPC64LoweredAtomicAdd32 {
			p5 := gc.Prog(ppc64.AMOVWZ)
			p5.To.Type = obj.TYPE_REG
			p5.To.Reg = out
			p5.From.Type = obj.TYPE_REG
			p5.From.Reg = out
		}

	case ssa.OpPPC64LoweredAtomicExchange32,
		ssa.OpPPC64LoweredAtomicExchange64:
		// SYNC
		// LDAR/LWAR    (Rarg0), Rout
		// STDCCC/STWCCC Rout, (Rarg0)
		// BNE         -2(PC)
		// ISYNC
		ld := ppc64.ALDAR
		st := ppc64.ASTDCCC
		if v.Op == ssa.OpPPC64LoweredAtomicExchange32 {
			ld = ppc64.ALWAR
			st = ppc64.ASTWCCC
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		// SYNC
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		// LDAR or LWAR
		p := gc.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		// STDCCC or STWCCC
		p1 := gc.Prog(st)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		// BNE retry
		p2 := gc.Prog(ppc64.ABNE)
		p2.To.Type = obj.TYPE_BRANCH
		gc.Patch(p2, p)
		// ISYNC
		pisync := gc.Prog(ppc64.AISYNC)
		pisync.To.Type = obj.TYPE_NONE

	case ssa.OpPPC64LoweredAtomicLoad32,
		ssa.OpPPC64LoweredAtomicLoad64,
		ssa.OpPPC64LoweredAtomicLoadPtr:
		// SYNC
		// MOVD/MOVW (Rarg0), Rout
		// CMP Rout,Rout
		// BNE 1(PC)
		// ISYNC
		ld := ppc64.AMOVD
		cmp := ppc64.ACMP
		if v.Op == ssa.OpPPC64LoweredAtomicLoad32 {
			ld = ppc64.AMOVW
			cmp = ppc64.ACMPW
		}
		arg0 := v.Args[0].Reg()
		out := v.Reg0()
		// SYNC
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		// Load
		p := gc.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = arg0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		// CMP
		p1 := gc.Prog(cmp)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = out
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out
		// BNE
		p2 := gc.Prog(ppc64.ABNE)
		p2.To.Type = obj.TYPE_BRANCH
		// ISYNC
		pisync := gc.Prog(ppc64.AISYNC)
		pisync.To.Type = obj.TYPE_NONE
		gc.Patch(p2, pisync)

	case ssa.OpPPC64LoweredAtomicStore32,
		ssa.OpPPC64LoweredAtomicStore64:
		// SYNC
		// MOVD/MOVW arg1,(arg0)
		st := ppc64.AMOVD
		if v.Op == ssa.OpPPC64LoweredAtomicStore32 {
			st = ppc64.AMOVW
		}
		arg0 := v.Args[0].Reg()
		arg1 := v.Args[1].Reg()
		// SYNC
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		// Store
		p := gc.Prog(st)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arg0
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arg1

	case ssa.OpPPC64LoweredAtomicCas64,
		ssa.OpPPC64LoweredAtomicCas32:
		// SYNC
		// loop:
		// LDAR        (Rarg0), Rtmp
		// CMP         Rarg1, Rtmp
		// BNE         fail
		// STDCCC      Rarg2, (Rarg0)
		// BNE         loop
		// ISYNC
		// MOVD        $1, Rout
		// BR          end
		// fail:
		// MOVD        $0, Rout
		// end:
		ld := ppc64.ALDAR
		st := ppc64.ASTDCCC
		cmp := ppc64.ACMP
		if v.Op == ssa.OpPPC64LoweredAtomicCas32 {
			ld = ppc64.ALWAR
			st = ppc64.ASTWCCC
			cmp = ppc64.ACMPW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()
		// SYNC
		psync := gc.Prog(ppc64.ASYNC)
		psync.To.Type = obj.TYPE_NONE
		// LDAR or LWAR
		p := gc.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP
		// CMP reg1,reg2
		p1 := gc.Prog(cmp)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Reg = ppc64.REGTMP
		p1.To.Type = obj.TYPE_REG
		// BNE cas_fail
		p2 := gc.Prog(ppc64.ABNE)
		p2.To.Type = obj.TYPE_BRANCH
		// STDCCC or STWCCC
		p3 := gc.Prog(st)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = r2
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = r0
		// BNE retry
		p4 := gc.Prog(ppc64.ABNE)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)
		// ISYNC
		pisync := gc.Prog(ppc64.AISYNC)
		pisync.To.Type = obj.TYPE_NONE
		// return true
		p5 := gc.Prog(ppc64.AMOVD)
		p5.From.Type = obj.TYPE_CONST
		p5.From.Offset = 1
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = out
		// BR done
		p6 := gc.Prog(obj.AJMP)
		p6.To.Type = obj.TYPE_BRANCH
		// return false
		p7 := gc.Prog(ppc64.AMOVD)
		p7.From.Type = obj.TYPE_CONST
		p7.From.Offset = 0
		p7.To.Type = obj.TYPE_REG
		p7.To.Reg = out
		gc.Patch(p2, p7)
		// done (label)
		p8 := gc.Prog(obj.ANOP)
		gc.Patch(p6, p8)

	case ssa.OpPPC64LoweredGetClosurePtr:
		// Closure pointer is R11 (already)
		gc.CheckLoweredGetClosurePtr(v)

	case ssa.OpLoadReg:
		loadOp := loadByType(v.Type)
		p := gc.Prog(loadOp)
		gc.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpStoreReg:
		storeOp := storeByType(v.Type)
		p := gc.Prog(storeOp)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		gc.AddrAuto(&p.To, v)

	case ssa.OpPPC64DIVD:
		// For now,
		//
		// cmp arg1, -1
		// be  ahead
		// v = arg0 / arg1
		// b over
		// ahead: v = - arg0
		// over: nop
		r := v.Reg()
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()

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
		r := v.Reg()
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()

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
		ssa.OpPPC64AND, ssa.OpPPC64OR, ssa.OpPPC64ANDN, ssa.OpPPC64ORN, ssa.OpPPC64NOR, ssa.OpPPC64XOR, ssa.OpPPC64EQV:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.OpPPC64MaskIfNotCarry:
		r := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssa.OpPPC64ADDconstForCarry:
		r1 := v.Args[0].Reg()
		p := gc.Prog(v.Op.Asm())
		p.Reg = r1
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP // Ignored; this is for the carry effect.

	case ssa.OpPPC64NEG, ssa.OpPPC64FNEG, ssa.OpPPC64FSQRT, ssa.OpPPC64FSQRTS, ssa.OpPPC64FCTIDZ, ssa.OpPPC64FCTIWZ, ssa.OpPPC64FCFID, ssa.OpPPC64FRSP:
		r := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()

	case ssa.OpPPC64ADDconst, ssa.OpPPC64ANDconst, ssa.OpPPC64ORconst, ssa.OpPPC64XORconst,
		ssa.OpPPC64SRADconst, ssa.OpPPC64SRAWconst, ssa.OpPPC64SRDconst, ssa.OpPPC64SRWconst, ssa.OpPPC64SLDconst, ssa.OpPPC64SLWconst:
		p := gc.Prog(v.Op.Asm())
		p.Reg = v.Args[0].Reg()

		if v.Aux != nil {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = gc.AuxOffset(v)
		} else {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = v.AuxInt
		}

		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpPPC64ANDCCconst:
		p := gc.Prog(v.Op.Asm())
		p.Reg = v.Args[0].Reg()

		if v.Aux != nil {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = gc.AuxOffset(v)
		} else {
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = v.AuxInt
		}

		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP // discard result

	case ssa.OpPPC64MOVDaddr:
		p := gc.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

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
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}

	case ssa.OpPPC64MOVDconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpPPC64FMOVDconst, ssa.OpPPC64FMOVSconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpPPC64FCMPU, ssa.OpPPC64CMP, ssa.OpPPC64CMPW, ssa.OpPPC64CMPU, ssa.OpPPC64CMPWU:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[1].Reg()

	case ssa.OpPPC64CMPconst, ssa.OpPPC64CMPUconst, ssa.OpPPC64CMPWconst, ssa.OpPPC64CMPWUconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt

	case ssa.OpPPC64MOVBreg, ssa.OpPPC64MOVBZreg, ssa.OpPPC64MOVHreg, ssa.OpPPC64MOVHZreg, ssa.OpPPC64MOVWreg, ssa.OpPPC64MOVWZreg:
		// Shift in register to required size
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Reg = v.Reg()
		p.To.Type = obj.TYPE_REG

	case ssa.OpPPC64MOVDload, ssa.OpPPC64MOVWload, ssa.OpPPC64MOVHload, ssa.OpPPC64MOVWZload, ssa.OpPPC64MOVBZload, ssa.OpPPC64MOVHZload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpPPC64FMOVDload, ssa.OpPPC64FMOVSload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpPPC64MOVDstorezero, ssa.OpPPC64MOVWstorezero, ssa.OpPPC64MOVHstorezero, ssa.OpPPC64MOVBstorezero:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)

	case ssa.OpPPC64MOVDstore, ssa.OpPPC64MOVWstore, ssa.OpPPC64MOVHstore, ssa.OpPPC64MOVBstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpPPC64FMOVDstore, ssa.OpPPC64FMOVSstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
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
		//   rtmp := 1
		//   isel rt,rtmp,r0,cond // rt is target in ppc asm

		// for  a >= b, a <= b, a != b:
		//   rtmp := 1
		//   isel rt,0,rtmp,!cond // rt is target in ppc asm

		if v.Block.Func.Config.OldArch {
			p := gc.Prog(ppc64.AMOVD)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
			p.To.Type = obj.TYPE_REG
			p.To.Reg = v.Reg()

			pb := gc.Prog(condOps[v.Op])
			pb.To.Type = obj.TYPE_BRANCH

			p = gc.Prog(ppc64.AMOVD)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = v.Reg()

			p = gc.Prog(obj.ANOP)
			gc.Patch(pb, p)
			break
		}
		// Modern PPC uses ISEL
		p := gc.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = iselRegs[1]
		iop := iselOps[v.Op]
		ssaGenISEL(v, iop.cond, iselRegs[iop.valueIfCond], iselRegs[1-iop.valueIfCond])

	case ssa.OpPPC64FLessEqual, // These include a second branch for EQ -- dealing with NaN prevents REL= to !REL conversion
		ssa.OpPPC64FGreaterEqual:

		if v.Block.Func.Config.OldArch {
			p := gc.Prog(ppc64.AMOVW)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
			p.To.Type = obj.TYPE_REG
			p.To.Reg = v.Reg()

			pb0 := gc.Prog(condOps[v.Op])
			pb0.To.Type = obj.TYPE_BRANCH
			pb1 := gc.Prog(ppc64.ABEQ)
			pb1.To.Type = obj.TYPE_BRANCH

			p = gc.Prog(ppc64.AMOVW)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 0
			p.To.Type = obj.TYPE_REG
			p.To.Reg = v.Reg()

			p = gc.Prog(obj.ANOP)
			gc.Patch(pb0, p)
			gc.Patch(pb1, p)
			break
		}
		// Modern PPC uses ISEL
		p := gc.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = iselRegs[1]
		iop := iselOps[v.Op]
		ssaGenISEL(v, iop.cond, iselRegs[iop.valueIfCond], iselRegs[1-iop.valueIfCond])
		ssaGenISEL(v, ppc64.C_COND_EQ, iselRegs[1], v.Reg())

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
		p.Reg = v.Args[0].Reg()
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()

		p = gc.Prog(movu)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ppc64.REG_R0
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = sz

		p2 := gc.Prog(ppc64.ACMPU)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[0].Reg()
		p2.To.Reg = v.Args[1].Reg()
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
		p.Reg = v.Args[0].Reg()
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[0].Reg()

		p = gc.Prog(ppc64.AADD)
		p.Reg = v.Args[1].Reg()
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = -sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Args[1].Reg()

		p = gc.Prog(movu)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP

		p2 := gc.Prog(movu)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = ppc64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p2.To.Offset = sz

		p3 := gc.Prog(ppc64.ACMPU)
		p3.From.Reg = v.Args[1].Reg()
		p3.From.Type = obj.TYPE_REG
		p3.To.Reg = v.Args[2].Reg()
		p3.To.Type = obj.TYPE_REG

		p4 := gc.Prog(ppc64.ABLT)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)

	case ssa.OpPPC64CALLstatic:
		if v.Aux.(*obj.LSym) == gc.Deferreturn {
			// Deferred calls will appear to be returning to
			// the CALL deferreturn(SB) that we are about to emit.
			// However, the stack trace code will show the line
			// of the instruction byte before the return PC.
			// To avoid that being an unrelated instruction,
			// insert two actual hardware NOPs that will have the right line number.
			// This is different from obj.ANOP, which is a virtual no-op
			// that doesn't make it into the instruction stream.
			// PPC64 is unusual because TWO nops are required
			// (see gc/cgen.go, gc/plive.go -- copy of comment below)
			//
			// On ppc64, when compiling Go into position
			// independent code on ppc64le we insert an
			// instruction to reload the TOC pointer from the
			// stack as well. See the long comment near
			// jmpdefer in runtime/asm_ppc64.s for why.
			// If the MOVD is not needed, insert a hardware NOP
			// so that the same number of instructions are used
			// on ppc64 in both shared and non-shared modes.
			ginsnop()
			if gc.Ctxt.Flag_shared {
				p := gc.Prog(ppc64.AMOVD)
				p.From.Type = obj.TYPE_MEM
				p.From.Offset = 24
				p.From.Reg = ppc64.REGSP
				p.To.Type = obj.TYPE_REG
				p.To.Reg = ppc64.REG_R2
			} else {
				ginsnop()
			}
		}
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = v.Aux.(*obj.LSym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}

	case ssa.OpPPC64CALLclosure, ssa.OpPPC64CALLinter:
		p := gc.Prog(ppc64.AMOVD)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REG_CTR

		if gc.Ctxt.Flag_shared && p.From.Reg != ppc64.REG_R12 {
			// Make sure function pointer is in R12 as well when
			// compiling Go into PIC.
			// TODO(mwhudson): it would obviously be better to
			// change the register allocation to put the value in
			// R12 already, but I don't know how to do that.
			// TODO: We have the technology now to implement TODO above.
			q := gc.Prog(ppc64.AMOVD)
			q.From = p.From
			q.To.Type = obj.TYPE_REG
			q.To.Reg = ppc64.REG_R12
		}

		pp := gc.Prog(obj.ACALL)
		pp.To.Type = obj.TYPE_REG
		pp.To.Reg = ppc64.REG_CTR

		if gc.Ctxt.Flag_shared {
			// When compiling Go into PIC, the function we just
			// called via pointer might have been implemented in
			// a separate module and so overwritten the TOC
			// pointer in R2; reload it.
			q := gc.Prog(ppc64.AMOVD)
			q.From.Type = obj.TYPE_MEM
			q.From.Offset = 24
			q.From.Reg = ppc64.REGSP
			q.To.Type = obj.TYPE_REG
			q.To.Reg = ppc64.REG_R2
		}

		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}

	case ssa.OpPPC64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := gc.Prog(ppc64.AMOVBZ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ppc64.REGTMP
		if gc.Debug_checknil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			gc.Warnl(v.Pos, "generated nil check")
		}

	case ssa.OpPPC64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.OpPPC64FlagEQ, ssa.OpPPC64FlagLT, ssa.OpPPC64FlagGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())

	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = [...]struct {
	asm, invasm     obj.As
	asmeq, invasmun bool
}{
	ssa.BlockPPC64EQ: {ppc64.ABEQ, ppc64.ABNE, false, false},
	ssa.BlockPPC64NE: {ppc64.ABNE, ppc64.ABEQ, false, false},

	ssa.BlockPPC64LT: {ppc64.ABLT, ppc64.ABGE, false, false},
	ssa.BlockPPC64GE: {ppc64.ABGE, ppc64.ABLT, false, false},
	ssa.BlockPPC64LE: {ppc64.ABLE, ppc64.ABGT, false, false},
	ssa.BlockPPC64GT: {ppc64.ABGT, ppc64.ABLE, false, false},

	// TODO: need to work FP comparisons into block jumps
	ssa.BlockPPC64FLT: {ppc64.ABLT, ppc64.ABGE, false, false},
	ssa.BlockPPC64FGE: {ppc64.ABGT, ppc64.ABLT, true, true}, // GE = GT or EQ; !GE = LT or UN
	ssa.BlockPPC64FLE: {ppc64.ABLT, ppc64.ABGT, true, true}, // LE = LT or EQ; !LE = GT or UN
	ssa.BlockPPC64FGT: {ppc64.ABGT, ppc64.ABLE, false, false},
}

func ssaGenBlock(s *gc.SSAGenState, b, next *ssa.Block) {
	s.SetPos(b.Pos)

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

	case ssa.BlockPlain:
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
		p.To.Sym = b.Aux.(*obj.LSym)

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
			if jmp.invasmun {
				// TODO: The second branch is probably predict-not-taken since it is for FP unordered
				q := gc.Prog(ppc64.ABVS)
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
		b.Fatalf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
