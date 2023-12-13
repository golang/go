// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/loong64"
)

// isFPreg reports whether r is an FP register.
func isFPreg(r int16) bool {
	return loong64.REG_F0 <= r && r <= loong64.REG_F31
}

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type, r int16) obj.As {
	if isFPreg(r) {
		if t.Size() == 4 {
			return loong64.AMOVF
		} else {
			return loong64.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return loong64.AMOVB
			} else {
				return loong64.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return loong64.AMOVH
			} else {
				return loong64.AMOVHU
			}
		case 4:
			if t.IsSigned() {
				return loong64.AMOVW
			} else {
				return loong64.AMOVWU
			}
		case 8:
			return loong64.AMOVV
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type, r int16) obj.As {
	if isFPreg(r) {
		if t.Size() == 4 {
			return loong64.AMOVF
		} else {
			return loong64.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return loong64.AMOVB
		case 2:
			return loong64.AMOVH
		case 4:
			return loong64.AMOVW
		case 8:
			return loong64.AMOVV
		}
	}
	panic("bad store type")
}

// largestMove returns the largest move instruction possible and its size,
// given the alignment of the total size of the move.
//
// e.g., a 16-byte move may use MOVV, but an 11-byte move must use MOVB.
//
// Note that the moves may not be on naturally aligned addresses depending on
// the source and destination.
//
// This matches the calculation in ssa.moveSize.
func largestMove(alignment int64) (obj.As, int64) {
	switch {
	case alignment%8 == 0:
		return loong64.AMOVV, 8
	case alignment%4 == 0:
		return loong64.AMOVW, 4
	case alignment%2 == 0:
		return loong64.AMOVH, 2
	default:
		return loong64.AMOVB, 1
	}
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssa.OpCopy, ssa.OpLOONG64MOVVreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := loong64.AMOVV
		if isFPreg(x) && isFPreg(y) {
			as = loong64.AMOVD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssa.OpLOONG64MOVVnop:
		// nothing to do
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		r := v.Reg()
		p := s.Prog(loadByType(v.Type, r))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		p := s.Prog(storeByType(v.Type, r))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		ssagen.AddrAuto(&p.To, v)
	case ssa.OpArgIntReg, ssa.OpArgFloatReg:
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		for _, a := range v.Block.Func.RegArgs {
			// Pass the spill/unspill information along to the assembler, offset by size of
			// the saved LR slot.
			addr := ssagen.SpillSlotAddr(a, loong64.REGSP, base.Ctxt.Arch.FixedFrameSize)
			s.FuncInfo().AddSpill(
				obj.RegSpill{Reg: a.Reg, Addr: addr, Unspill: loadByType(a.Type, a.Reg), Spill: storeByType(a.Type, a.Reg)})
		}
		v.Block.Func.RegArgs = nil
		ssagen.CheckArgReg(v)
	case ssa.OpLOONG64ADDV,
		ssa.OpLOONG64SUBV,
		ssa.OpLOONG64AND,
		ssa.OpLOONG64OR,
		ssa.OpLOONG64XOR,
		ssa.OpLOONG64NOR,
		ssa.OpLOONG64SLLV,
		ssa.OpLOONG64SRLV,
		ssa.OpLOONG64SRAV,
		ssa.OpLOONG64ROTR,
		ssa.OpLOONG64ROTRV,
		ssa.OpLOONG64ADDF,
		ssa.OpLOONG64ADDD,
		ssa.OpLOONG64SUBF,
		ssa.OpLOONG64SUBD,
		ssa.OpLOONG64MULF,
		ssa.OpLOONG64MULD,
		ssa.OpLOONG64DIVF,
		ssa.OpLOONG64DIVD,
		ssa.OpLOONG64MULV, ssa.OpLOONG64MULHV, ssa.OpLOONG64MULHVU,
		ssa.OpLOONG64DIVV, ssa.OpLOONG64REMV, ssa.OpLOONG64DIVVU, ssa.OpLOONG64REMVU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64SGT,
		ssa.OpLOONG64SGTU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64ADDVconst,
		ssa.OpLOONG64SUBVconst,
		ssa.OpLOONG64ANDconst,
		ssa.OpLOONG64ORconst,
		ssa.OpLOONG64XORconst,
		ssa.OpLOONG64NORconst,
		ssa.OpLOONG64SLLVconst,
		ssa.OpLOONG64SRLVconst,
		ssa.OpLOONG64SRAVconst,
		ssa.OpLOONG64ROTRconst,
		ssa.OpLOONG64ROTRVconst,
		ssa.OpLOONG64SGTconst,
		ssa.OpLOONG64SGTUconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64MOVVconst:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if isFPreg(r) {
			// cannot move into FP or special registers, use TMP as intermediate
			p.To.Reg = loong64.REGTMP
			p = s.Prog(loong64.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = loong64.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpLOONG64MOVFconst,
		ssa.OpLOONG64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64CMPEQF,
		ssa.OpLOONG64CMPEQD,
		ssa.OpLOONG64CMPGEF,
		ssa.OpLOONG64CMPGED,
		ssa.OpLOONG64CMPGTF,
		ssa.OpLOONG64CMPGTD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
	case ssa.OpLOONG64MOVVaddr:
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		var wantreg string
		// MOVV $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R3)
		// when constant is large, tmp register (R30) may be used
		// - base is SB: load external address with relocation
		switch v.Aux.(type) {
		default:
			v.Fatalf("aux is of unknown type %T", v.Aux)
		case *obj.LSym:
			wantreg = "SB"
			ssagen.AddAux(&p.From, v)
		case *ir.Name:
			wantreg = "SP"
			ssagen.AddAux(&p.From, v)
		case nil:
			// No sym, just MOVV $off(SP), R
			wantreg = "SP"
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64MOVBload,
		ssa.OpLOONG64MOVBUload,
		ssa.OpLOONG64MOVHload,
		ssa.OpLOONG64MOVHUload,
		ssa.OpLOONG64MOVWload,
		ssa.OpLOONG64MOVWUload,
		ssa.OpLOONG64MOVVload,
		ssa.OpLOONG64MOVFload,
		ssa.OpLOONG64MOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64MOVBstore,
		ssa.OpLOONG64MOVHstore,
		ssa.OpLOONG64MOVWstore,
		ssa.OpLOONG64MOVVstore,
		ssa.OpLOONG64MOVFstore,
		ssa.OpLOONG64MOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpLOONG64MOVBstorezero,
		ssa.OpLOONG64MOVHstorezero,
		ssa.OpLOONG64MOVWstorezero,
		ssa.OpLOONG64MOVVstorezero:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpLOONG64MOVBreg,
		ssa.OpLOONG64MOVBUreg,
		ssa.OpLOONG64MOVHreg,
		ssa.OpLOONG64MOVHUreg,
		ssa.OpLOONG64MOVWreg,
		ssa.OpLOONG64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpLOONG64MOVVreg {
			a = a.Args[0]
		}
		if a.Op == ssa.OpLoadReg && loong64.REG_R0 <= a.Reg() && a.Reg() <= loong64.REG_R31 {
			// LoadReg from a narrower type does an extension, except loading
			// to a floating point register. So only eliminate the extension
			// if it is loaded to an integer register.

			t := a.Type
			switch {
			case v.Op == ssa.OpLOONG64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpLOONG64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpLOONG64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpLOONG64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssa.OpLOONG64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssa.OpLOONG64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := s.Prog(loong64.AMOVV)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssa.OpLOONG64MOVWF,
		ssa.OpLOONG64MOVWD,
		ssa.OpLOONG64TRUNCFW,
		ssa.OpLOONG64TRUNCDW,
		ssa.OpLOONG64MOVVF,
		ssa.OpLOONG64MOVVD,
		ssa.OpLOONG64TRUNCFV,
		ssa.OpLOONG64TRUNCDV,
		ssa.OpLOONG64MOVFD,
		ssa.OpLOONG64MOVDF,
		ssa.OpLOONG64NEGF,
		ssa.OpLOONG64NEGD,
		ssa.OpLOONG64SQRTD,
		ssa.OpLOONG64SQRTF:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64NEGV:
		// SUB from REGZERO
		p := s.Prog(loong64.ASUBVU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64DUFFZERO:
		// runtime.duffzero expects start address in R20
		p := s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt
	case ssa.OpLOONG64LoweredZero:
		// MOVx	R0, (Rarg0)
		// ADDV	$sz, Rarg0
		// BGEU	Rarg1, Rarg0, -2(PC)
		mov, sz := largestMove(v.AuxInt)
		p := s.Prog(mov)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()

		p2 := s.Prog(loong64.AADDVU)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = sz
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Args[0].Reg()

		p3 := s.Prog(loong64.ABGEU)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[1].Reg()
		p3.Reg = v.Args[0].Reg()
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)

	case ssa.OpLOONG64DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt
	case ssa.OpLOONG64LoweredMove:
		// MOVx	(Rarg1), Rtmp
		// MOVx	Rtmp, (Rarg0)
		// ADDV	$sz, Rarg1
		// ADDV	$sz, Rarg0
		// BGEU	Rarg2, Rarg0, -4(PC)
		mov, sz := largestMove(v.AuxInt)
		p := s.Prog(mov)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REGTMP

		p2 := s.Prog(mov)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = loong64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := s.Prog(loong64.AADDVU)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = sz
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Args[1].Reg()

		p4 := s.Prog(loong64.AADDVU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = sz
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Args[0].Reg()

		p5 := s.Prog(loong64.ABGEU)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Args[2].Reg()
		p5.Reg = v.Args[1].Reg()
		p5.To.Type = obj.TYPE_BRANCH
		p5.To.SetTarget(p)

	case ssa.OpLOONG64CALLstatic, ssa.OpLOONG64CALLclosure, ssa.OpLOONG64CALLinter:
		s.Call(v)
	case ssa.OpLOONG64CALLtail:
		s.TailCall(v)
	case ssa.OpLOONG64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]
	case ssa.OpLOONG64LoweredPanicBoundsA, ssa.OpLOONG64LoweredPanicBoundsB, ssa.OpLOONG64LoweredPanicBoundsC:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ssagen.BoundsCheckFunc[v.AuxInt]
		s.UseArgs(16) // space used in callee args area by assembly stubs
	case ssa.OpLOONG64LoweredAtomicLoad8, ssa.OpLOONG64LoweredAtomicLoad32, ssa.OpLOONG64LoweredAtomicLoad64:
		as := loong64.AMOVV
		switch v.Op {
		case ssa.OpLOONG64LoweredAtomicLoad8:
			as = loong64.AMOVB
		case ssa.OpLOONG64LoweredAtomicLoad32:
			as = loong64.AMOVW
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		s.Prog(loong64.ADBAR)
	case ssa.OpLOONG64LoweredAtomicStore8, ssa.OpLOONG64LoweredAtomicStore32, ssa.OpLOONG64LoweredAtomicStore64:
		as := loong64.AMOVV
		switch v.Op {
		case ssa.OpLOONG64LoweredAtomicStore8:
			as = loong64.AMOVB
		case ssa.OpLOONG64LoweredAtomicStore32:
			as = loong64.AMOVW
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		s.Prog(loong64.ADBAR)
	case ssa.OpLOONG64LoweredAtomicStorezero32, ssa.OpLOONG64LoweredAtomicStorezero64:
		as := loong64.AMOVV
		if v.Op == ssa.OpLOONG64LoweredAtomicStorezero32 {
			as = loong64.AMOVW
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		s.Prog(loong64.ADBAR)
	case ssa.OpLOONG64LoweredAtomicExchange32, ssa.OpLOONG64LoweredAtomicExchange64:
		// DBAR
		// MOVV	Rarg1, Rtmp
		// LL	(Rarg0), Rout
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// DBAR
		ll := loong64.ALLV
		sc := loong64.ASCV
		if v.Op == ssa.OpLOONG64LoweredAtomicExchange32 {
			ll = loong64.ALL
			sc = loong64.ASC
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REGTMP
		p1 := s.Prog(ll)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = loong64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(loong64.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = loong64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(loong64.ADBAR)
	case ssa.OpLOONG64LoweredAtomicAdd32, ssa.OpLOONG64LoweredAtomicAdd64:
		// DBAR
		// LL	(Rarg0), Rout
		// ADDV Rarg1, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// DBAR
		// ADDV Rarg1, Rout
		ll := loong64.ALLV
		sc := loong64.ASCV
		if v.Op == ssa.OpLOONG64LoweredAtomicAdd32 {
			ll = loong64.ALL
			sc = loong64.ASC
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(ll)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(loong64.AADDVU)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = loong64.REGTMP
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = loong64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(loong64.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = loong64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(loong64.ADBAR)
		p4 := s.Prog(loong64.AADDVU)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[1].Reg()
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()
	case ssa.OpLOONG64LoweredAtomicAddconst32, ssa.OpLOONG64LoweredAtomicAddconst64:
		// DBAR
		// LL	(Rarg0), Rout
		// ADDV $auxint, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// DBAR
		// ADDV $auxint, Rout
		ll := loong64.ALLV
		sc := loong64.ASCV
		if v.Op == ssa.OpLOONG64LoweredAtomicAddconst32 {
			ll = loong64.ALL
			sc = loong64.ASC
		}
		s.Prog(loong64.ADBAR)
		p := s.Prog(ll)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(loong64.AADDVU)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = v.AuxInt
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = loong64.REGTMP
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = loong64.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(loong64.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = loong64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(loong64.ADBAR)
		p4 := s.Prog(loong64.AADDVU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = v.AuxInt
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()
	case ssa.OpLOONG64LoweredAtomicCas32, ssa.OpLOONG64LoweredAtomicCas64:
		// MOVV $0, Rout
		// DBAR
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVV Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// DBAR
		ll := loong64.ALLV
		sc := loong64.ASCV
		if v.Op == ssa.OpLOONG64LoweredAtomicCas32 {
			ll = loong64.ALL
			sc = loong64.ASC
		}
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		s.Prog(loong64.ADBAR)
		p1 := s.Prog(ll)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = loong64.REGTMP
		p2 := s.Prog(loong64.ABNE)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = loong64.REGTMP
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(loong64.AMOVV)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg0()
		p4 := s.Prog(sc)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_MEM
		p4.To.Reg = v.Args[0].Reg()
		p5 := s.Prog(loong64.ABEQ)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Reg0()
		p5.To.Type = obj.TYPE_BRANCH
		p5.To.SetTarget(p1)
		p6 := s.Prog(loong64.ADBAR)
		p2.To.SetTarget(p6)
	case ssa.OpLOONG64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(loong64.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssa.OpLOONG64FPFlagTrue,
		ssa.OpLOONG64FPFlagFalse:
		// MOVV	$0, r
		// BFPF	2(PC)
		// MOVV	$1, r
		branch := loong64.ABFPF
		if v.Op == ssa.OpLOONG64FPFlagFalse {
			branch = loong64.ABFPT
		}
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p2 := s.Prog(branch)
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(loong64.AMOVV)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = 1
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg()
		p4 := s.Prog(obj.ANOP) // not a machine instruction, for branch to land
		p2.To.SetTarget(p4)
	case ssa.OpLOONG64LoweredGetClosurePtr:
		// Closure pointer is R22 (loong64.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssa.OpLOONG64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64MASKEQZ, ssa.OpLOONG64MASKNEZ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpClobber, ssa.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = map[ssa.BlockKind]struct {
	asm, invasm obj.As
}{
	ssa.BlockLOONG64EQ:  {loong64.ABEQ, loong64.ABNE},
	ssa.BlockLOONG64NE:  {loong64.ABNE, loong64.ABEQ},
	ssa.BlockLOONG64LTZ: {loong64.ABLTZ, loong64.ABGEZ},
	ssa.BlockLOONG64GEZ: {loong64.ABGEZ, loong64.ABLTZ},
	ssa.BlockLOONG64LEZ: {loong64.ABLEZ, loong64.ABGTZ},
	ssa.BlockLOONG64GTZ: {loong64.ABGTZ, loong64.ABLEZ},
	ssa.BlockLOONG64FPT: {loong64.ABFPT, loong64.ABFPF},
	ssa.BlockLOONG64FPF: {loong64.ABFPF, loong64.ABFPT},
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	switch b.Kind {
	case ssa.BlockPlain:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockDefer:
		// defer returns in R19:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := s.Prog(loong64.ABNE)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.Reg = loong64.REG_R19
		p.To.Type = obj.TYPE_BRANCH
		s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[1].Block()})
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockExit, ssa.BlockRetJmp:
	case ssa.BlockRet:
		s.Prog(obj.ARET)
	case ssa.BlockLOONG64EQ, ssa.BlockLOONG64NE,
		ssa.BlockLOONG64LTZ, ssa.BlockLOONG64GEZ,
		ssa.BlockLOONG64LEZ, ssa.BlockLOONG64GTZ,
		ssa.BlockLOONG64FPT, ssa.BlockLOONG64FPF:
		jmp := blockJump[b.Kind]
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			p = s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				p = s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				p = s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}
		if !b.Controls[0].Type.IsFlags() {
			p.From.Type = obj.TYPE_REG
			p.From.Reg = b.Controls[0].Reg()
		}
	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}

func loadRegResult(s *ssagen.State, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p := s.Prog(loadByType(t, reg))
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_AUTO
	p.From.Sym = n.Linksym()
	p.From.Offset = n.FrameOffset() + off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = reg
	return p
}

func spillArgReg(pp *objw.Progs, p *obj.Prog, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p = pp.Append(p, storeByType(t, reg), obj.TYPE_REG, reg, 0, obj.TYPE_MEM, 0, n.FrameOffset()+off)
	p.To.Name = obj.NAME_PARAM
	p.To.Sym = n.Linksym()
	p.Pos = p.Pos.WithNotStmt()
	return p
}
