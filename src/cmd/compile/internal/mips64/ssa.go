// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"math"

	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

// isFPreg returns whether r is an FP register
func isFPreg(r int16) bool {
	return mips.REG_F0 <= r && r <= mips.REG_F31
}

// isHILO returns whether r is HI or LO register
func isHILO(r int16) bool {
	return r == mips.REG_HI || r == mips.REG_LO
}

// loadByType returns the load instruction of the given type.
func loadByType(t ssa.Type, r int16) obj.As {
	if isFPreg(r) {
		if t.Size() == 4 { // float32 or int32
			return mips.AMOVF
		} else { // float64 or int64
			return mips.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return mips.AMOVB
			} else {
				return mips.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return mips.AMOVH
			} else {
				return mips.AMOVHU
			}
		case 4:
			if t.IsSigned() {
				return mips.AMOVW
			} else {
				return mips.AMOVWU
			}
		case 8:
			return mips.AMOVV
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t ssa.Type, r int16) obj.As {
	if isFPreg(r) {
		if t.Size() == 4 { // float32 or int32
			return mips.AMOVF
		} else { // float64 or int64
			return mips.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return mips.AMOVB
		case 2:
			return mips.AMOVH
		case 4:
			return mips.AMOVW
		case 8:
			return mips.AMOVV
		}
	}
	panic("bad store type")
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
	case ssa.OpCopy, ssa.OpMIPS64MOVVconvert, ssa.OpMIPS64MOVVreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := mips.AMOVV
		if isFPreg(x) && isFPreg(y) {
			as = mips.AMOVD
		}
		p := gc.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
		if isHILO(x) && isHILO(y) || isHILO(x) && isFPreg(y) || isFPreg(x) && isHILO(y) {
			// cannot move between special registers, use TMP as intermediate
			p.To.Reg = mips.REGTMP
			p = gc.Prog(mips.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
		}
	case ssa.OpMIPS64MOVVnop:
		if v.Reg() != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		// nothing to do
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		r := v.Reg()
		p := gc.Prog(loadByType(v.Type, r))
		gc.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if isHILO(r) {
			// cannot directly load, load to TMP and move
			p.To.Reg = mips.REGTMP
			p = gc.Prog(mips.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpPhi:
		gc.CheckLoweredPhi(v)
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		if isHILO(r) {
			// cannot directly store, move to TMP and store
			p := gc.Prog(mips.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = r
			p.To.Type = obj.TYPE_REG
			p.To.Reg = mips.REGTMP
			r = mips.REGTMP
		}
		p := gc.Prog(storeByType(v.Type, r))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		gc.AddrAuto(&p.To, v)
	case ssa.OpMIPS64ADDV,
		ssa.OpMIPS64SUBV,
		ssa.OpMIPS64AND,
		ssa.OpMIPS64OR,
		ssa.OpMIPS64XOR,
		ssa.OpMIPS64NOR,
		ssa.OpMIPS64SLLV,
		ssa.OpMIPS64SRLV,
		ssa.OpMIPS64SRAV,
		ssa.OpMIPS64ADDF,
		ssa.OpMIPS64ADDD,
		ssa.OpMIPS64SUBF,
		ssa.OpMIPS64SUBD,
		ssa.OpMIPS64MULF,
		ssa.OpMIPS64MULD,
		ssa.OpMIPS64DIVF,
		ssa.OpMIPS64DIVD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64SGT,
		ssa.OpMIPS64SGTU:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64ADDVconst,
		ssa.OpMIPS64SUBVconst,
		ssa.OpMIPS64ANDconst,
		ssa.OpMIPS64ORconst,
		ssa.OpMIPS64XORconst,
		ssa.OpMIPS64NORconst,
		ssa.OpMIPS64SLLVconst,
		ssa.OpMIPS64SRLVconst,
		ssa.OpMIPS64SRAVconst,
		ssa.OpMIPS64SGTconst,
		ssa.OpMIPS64SGTUconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64MULV,
		ssa.OpMIPS64MULVU,
		ssa.OpMIPS64DIVV,
		ssa.OpMIPS64DIVVU:
		// result in hi,lo
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssa.OpMIPS64MOVVconst:
		r := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if isFPreg(r) || isHILO(r) {
			// cannot move into FP or special registers, use TMP as intermediate
			p.To.Reg = mips.REGTMP
			p = gc.Prog(mips.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpMIPS64MOVFconst,
		ssa.OpMIPS64MOVDconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64CMPEQF,
		ssa.OpMIPS64CMPEQD,
		ssa.OpMIPS64CMPGEF,
		ssa.OpMIPS64CMPGED,
		ssa.OpMIPS64CMPGTF,
		ssa.OpMIPS64CMPGTD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
	case ssa.OpMIPS64MOVVaddr:
		p := gc.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		var wantreg string
		// MOVV $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R29)
		//               when constant is large, tmp register (R23) may be used
		// - base is SB: load external address with relocation
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
			// No sym, just MOVV $off(SP), R
			wantreg = "SP"
			p.From.Reg = mips.REGSP
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64MOVBload,
		ssa.OpMIPS64MOVBUload,
		ssa.OpMIPS64MOVHload,
		ssa.OpMIPS64MOVHUload,
		ssa.OpMIPS64MOVWload,
		ssa.OpMIPS64MOVWUload,
		ssa.OpMIPS64MOVVload,
		ssa.OpMIPS64MOVFload,
		ssa.OpMIPS64MOVDload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64MOVBstore,
		ssa.OpMIPS64MOVHstore,
		ssa.OpMIPS64MOVWstore,
		ssa.OpMIPS64MOVVstore,
		ssa.OpMIPS64MOVFstore,
		ssa.OpMIPS64MOVDstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpMIPS64MOVBstorezero,
		ssa.OpMIPS64MOVHstorezero,
		ssa.OpMIPS64MOVWstorezero,
		ssa.OpMIPS64MOVVstorezero:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpMIPS64MOVBreg,
		ssa.OpMIPS64MOVBUreg,
		ssa.OpMIPS64MOVHreg,
		ssa.OpMIPS64MOVHUreg,
		ssa.OpMIPS64MOVWreg,
		ssa.OpMIPS64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpMIPS64MOVVreg {
			a = a.Args[0]
		}
		if a.Op == ssa.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssa.OpMIPS64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpMIPS64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpMIPS64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpMIPS64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssa.OpMIPS64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssa.OpMIPS64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := gc.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssa.OpMIPS64MOVWF,
		ssa.OpMIPS64MOVWD,
		ssa.OpMIPS64TRUNCFW,
		ssa.OpMIPS64TRUNCDW,
		ssa.OpMIPS64MOVVF,
		ssa.OpMIPS64MOVVD,
		ssa.OpMIPS64TRUNCFV,
		ssa.OpMIPS64TRUNCDV,
		ssa.OpMIPS64MOVFD,
		ssa.OpMIPS64MOVDF,
		ssa.OpMIPS64NEGF,
		ssa.OpMIPS64NEGD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64NEGV:
		// SUB from REGZERO
		p := gc.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPS64DUFFZERO:
		// runtime.duffzero expects start address - 8 in R1
		p := gc.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p = gc.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
		p.To.Offset = v.AuxInt
	case ssa.OpMIPS64LoweredZero:
		// SUBV	$8, R1
		// MOVV	R0, 8(R1)
		// ADDV	$8, R1
		// BNE	Rarg1, R1, -2(PC)
		// arg1 is the address of the last element to zero
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%8 == 0:
			sz = 8
			mov = mips.AMOVV
		case v.AuxInt%4 == 0:
			sz = 4
			mov = mips.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = mips.AMOVH
		default:
			sz = 1
			mov = mips.AMOVB
		}
		p := gc.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p2 := gc.Prog(mov)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGZERO
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = mips.REG_R1
		p2.To.Offset = sz
		p3 := gc.Prog(mips.AADDVU)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = sz
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = mips.REG_R1
		p4 := gc.Prog(mips.ABNE)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[1].Reg()
		p4.Reg = mips.REG_R1
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p2)
	case ssa.OpMIPS64LoweredMove:
		// SUBV	$8, R1
		// MOVV	8(R1), Rtmp
		// MOVV	Rtmp, (R2)
		// ADDV	$8, R1
		// ADDV	$8, R2
		// BNE	Rarg2, R1, -4(PC)
		// arg2 is the address of the last element of src
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%8 == 0:
			sz = 8
			mov = mips.AMOVV
		case v.AuxInt%4 == 0:
			sz = 4
			mov = mips.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = mips.AMOVH
		default:
			sz = 1
			mov = mips.AMOVB
		}
		p := gc.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p2 := gc.Prog(mov)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Reg = mips.REG_R1
		p2.From.Offset = sz
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = mips.REGTMP
		p3 := gc.Prog(mov)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = mips.REG_R2
		p4 := gc.Prog(mips.AADDVU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = sz
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = mips.REG_R1
		p5 := gc.Prog(mips.AADDVU)
		p5.From.Type = obj.TYPE_CONST
		p5.From.Offset = sz
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = mips.REG_R2
		p6 := gc.Prog(mips.ABNE)
		p6.From.Type = obj.TYPE_REG
		p6.From.Reg = v.Args[2].Reg()
		p6.Reg = mips.REG_R1
		p6.To.Type = obj.TYPE_BRANCH
		gc.Patch(p6, p2)
	case ssa.OpMIPS64CALLstatic:
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
	case ssa.OpMIPS64CALLclosure:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPS64CALLdefer:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Deferproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPS64CALLgo:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Newproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPS64CALLinter:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPS64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := gc.Prog(mips.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP
		if gc.Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			gc.Warnl(v.Line, "generated nil check")
		}
	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
	case ssa.OpKeepAlive:
		gc.KeepAlive(v)
	case ssa.OpMIPS64FPFlagTrue,
		ssa.OpMIPS64FPFlagFalse:
		// MOVV	$0, r
		// BFPF	2(PC)
		// MOVV	$1, r
		branch := mips.ABFPF
		if v.Op == ssa.OpMIPS64FPFlagFalse {
			branch = mips.ABFPT
		}
		p := gc.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p2 := gc.Prog(branch)
		p2.To.Type = obj.TYPE_BRANCH
		p3 := gc.Prog(mips.AMOVV)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = 1
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg()
		p4 := gc.Prog(obj.ANOP) // not a machine instruction, for branch to land
		gc.Patch(p2, p4)
	case ssa.OpSelect0, ssa.OpSelect1:
		// nothing to do
	case ssa.OpMIPS64LoweredGetClosurePtr:
		// Closure pointer is R22 (mips.REGCTXT).
		gc.CheckLoweredGetClosurePtr(v)
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = map[ssa.BlockKind]struct {
	asm, invasm obj.As
}{
	ssa.BlockMIPS64EQ:  {mips.ABEQ, mips.ABNE},
	ssa.BlockMIPS64NE:  {mips.ABNE, mips.ABEQ},
	ssa.BlockMIPS64LTZ: {mips.ABLTZ, mips.ABGEZ},
	ssa.BlockMIPS64GEZ: {mips.ABGEZ, mips.ABLTZ},
	ssa.BlockMIPS64LEZ: {mips.ABLEZ, mips.ABGTZ},
	ssa.BlockMIPS64GTZ: {mips.ABGTZ, mips.ABLEZ},
	ssa.BlockMIPS64FPT: {mips.ABFPT, mips.ABFPF},
	ssa.BlockMIPS64FPF: {mips.ABFPF, mips.ABFPT},
}

func ssaGenBlock(s *gc.SSAGenState, b, next *ssa.Block) {
	s.SetLineno(b.Line)

	switch b.Kind {
	case ssa.BlockPlain:
		if b.Succs[0].Block() != next {
			p := gc.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockDefer:
		// defer returns in R1:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := gc.Prog(mips.ABNE)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.Reg = mips.REG_R1
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
		p := gc.Prog(obj.ARET)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(b.Aux.(*gc.Sym))
	case ssa.BlockMIPS64EQ, ssa.BlockMIPS64NE,
		ssa.BlockMIPS64LTZ, ssa.BlockMIPS64GEZ,
		ssa.BlockMIPS64LEZ, ssa.BlockMIPS64GTZ,
		ssa.BlockMIPS64FPT, ssa.BlockMIPS64FPF:
		jmp := blockJump[b.Kind]
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = gc.Prog(jmp.invasm)
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
		if !b.Control.Type.IsFlags() {
			p.From.Type = obj.TYPE_REG
			p.From.Reg = b.Control.Reg()
		}
	default:
		b.Fatalf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
