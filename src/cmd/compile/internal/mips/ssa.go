// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips

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
			return mips.AMOVW
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
	case ssa.OpSelect0, ssa.OpSelect1:
		// nothing to do
	case ssa.OpCopy, ssa.OpMIPSMOVWconvert, ssa.OpMIPSMOVWreg:
		t := v.Type
		if t.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := mips.AMOVW
		if isFPreg(x) && isFPreg(y) {
			as = mips.AMOVF
			if t.Size() == 8 {
				as = mips.AMOVD
			}
		}

		p := gc.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
		if isHILO(x) && isHILO(y) || isHILO(x) && isFPreg(y) || isFPreg(x) && isHILO(y) {
			// cannot move between special registers, use TMP as intermediate
			p.To.Reg = mips.REGTMP
			p = gc.Prog(mips.AMOVW)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
		}
	case ssa.OpMIPSMOVWnop:
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
			p = gc.Prog(mips.AMOVW)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		if isHILO(r) {
			// cannot directly store, move to TMP and store
			p := gc.Prog(mips.AMOVW)
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
	case ssa.OpMIPSADD,
		ssa.OpMIPSSUB,
		ssa.OpMIPSAND,
		ssa.OpMIPSOR,
		ssa.OpMIPSXOR,
		ssa.OpMIPSNOR,
		ssa.OpMIPSSLL,
		ssa.OpMIPSSRL,
		ssa.OpMIPSSRA,
		ssa.OpMIPSADDF,
		ssa.OpMIPSADDD,
		ssa.OpMIPSSUBF,
		ssa.OpMIPSSUBD,
		ssa.OpMIPSMULF,
		ssa.OpMIPSMULD,
		ssa.OpMIPSDIVF,
		ssa.OpMIPSDIVD,
		ssa.OpMIPSMUL:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSSGT,
		ssa.OpMIPSSGTU:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSSGTzero,
		ssa.OpMIPSSGTUzero:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSADDconst,
		ssa.OpMIPSSUBconst,
		ssa.OpMIPSANDconst,
		ssa.OpMIPSORconst,
		ssa.OpMIPSXORconst,
		ssa.OpMIPSNORconst,
		ssa.OpMIPSSLLconst,
		ssa.OpMIPSSRLconst,
		ssa.OpMIPSSRAconst,
		ssa.OpMIPSSGTconst,
		ssa.OpMIPSSGTUconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSMULT,
		ssa.OpMIPSMULTU,
		ssa.OpMIPSDIV,
		ssa.OpMIPSDIVU:
		// result in hi,lo
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssa.OpMIPSMOVWconst:
		r := v.Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if isFPreg(r) || isHILO(r) {
			// cannot move into FP or special registers, use TMP as intermediate
			p.To.Reg = mips.REGTMP
			p = gc.Prog(mips.AMOVW)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpMIPSMOVFconst,
		ssa.OpMIPSMOVDconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSCMOVZ:
		if v.Reg() != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSCMOVZzero:
		if v.Reg() != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSCMPEQF,
		ssa.OpMIPSCMPEQD,
		ssa.OpMIPSCMPGEF,
		ssa.OpMIPSCMPGED,
		ssa.OpMIPSCMPGTF,
		ssa.OpMIPSCMPGTD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
	case ssa.OpMIPSMOVWaddr:
		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_ADDR
		var wantreg string
		// MOVW $sym+off(base), R
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
			// No sym, just MOVW $off(SP), R
			wantreg = "SP"
			p.From.Reg = mips.REGSP
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSMOVBload,
		ssa.OpMIPSMOVBUload,
		ssa.OpMIPSMOVHload,
		ssa.OpMIPSMOVHUload,
		ssa.OpMIPSMOVWload,
		ssa.OpMIPSMOVFload,
		ssa.OpMIPSMOVDload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSMOVBstore,
		ssa.OpMIPSMOVHstore,
		ssa.OpMIPSMOVWstore,
		ssa.OpMIPSMOVFstore,
		ssa.OpMIPSMOVDstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpMIPSMOVBstorezero,
		ssa.OpMIPSMOVHstorezero,
		ssa.OpMIPSMOVWstorezero:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpMIPSMOVBreg,
		ssa.OpMIPSMOVBUreg,
		ssa.OpMIPSMOVHreg,
		ssa.OpMIPSMOVHUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpMIPSMOVWreg || a.Op == ssa.OpMIPSMOVWnop {
			a = a.Args[0]
		}
		if a.Op == ssa.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssa.OpMIPSMOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpMIPSMOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpMIPSMOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpMIPSMOVHUreg && t.Size() == 2 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := gc.Prog(mips.AMOVW)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssa.OpMIPSMOVWF,
		ssa.OpMIPSMOVWD,
		ssa.OpMIPSTRUNCFW,
		ssa.OpMIPSTRUNCDW,
		ssa.OpMIPSMOVFD,
		ssa.OpMIPSMOVDF,
		ssa.OpMIPSNEGF,
		ssa.OpMIPSNEGD,
		ssa.OpMIPSSQRTD,
		ssa.OpMIPSCLZ:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSNEG:
		// SUB from REGZERO
		p := gc.Prog(mips.ASUBU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpMIPSLoweredZero:
		// SUBU	$4, R1
		// MOVW	R0, 4(R1)
		// ADDU	$4, R1
		// BNE	Rarg1, R1, -2(PC)
		// arg1 is the address of the last element to zero
		var sz int64
		var mov obj.As
		switch {
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
		p := gc.Prog(mips.ASUBU)
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
		p3 := gc.Prog(mips.AADDU)
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
	case ssa.OpMIPSLoweredMove:
		// SUBU	$4, R1
		// MOVW	4(R1), Rtmp
		// MOVW	Rtmp, (R2)
		// ADDU	$4, R1
		// ADDU	$4, R2
		// BNE	Rarg2, R1, -4(PC)
		// arg2 is the address of the last element of src
		var sz int64
		var mov obj.As
		switch {
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
		p := gc.Prog(mips.ASUBU)
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
		p4 := gc.Prog(mips.AADDU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = sz
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = mips.REG_R1
		p5 := gc.Prog(mips.AADDU)
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
	case ssa.OpMIPSCALLstatic:
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
	case ssa.OpMIPSCALLclosure:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPSCALLdefer:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Deferproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPSCALLgo:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Newproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPSCALLinter:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpMIPSLoweredAtomicLoad:
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		gc.Prog(mips.ASYNC)
	case ssa.OpMIPSLoweredAtomicStore:
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()

		gc.Prog(mips.ASYNC)
	case ssa.OpMIPSLoweredAtomicStorezero:
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()

		gc.Prog(mips.ASYNC)
	case ssa.OpMIPSLoweredAtomicExchange:
		// SYNC
		// MOVW Rarg1, Rtmp
		// LL	(Rarg0), Rout
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP

		p1 := gc.Prog(mips.ALL)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()

		p2 := gc.Prog(mips.ASC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := gc.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)

		gc.Prog(mips.ASYNC)
	case ssa.OpMIPSLoweredAtomicAdd:
		// SYNC
		// LL	(Rarg0), Rout
		// ADDU Rarg1, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDU Rarg1, Rout
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.ALL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		p1 := gc.Prog(mips.AADDU)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP

		p2 := gc.Prog(mips.ASC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := gc.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)

		gc.Prog(mips.ASYNC)

		p4 := gc.Prog(mips.AADDU)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[1].Reg()
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()

	case ssa.OpMIPSLoweredAtomicAddconst:
		// SYNC
		// LL	(Rarg0), Rout
		// ADDU $auxInt, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDU $auxInt, Rout
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.ALL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		p1 := gc.Prog(mips.AADDU)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = v.AuxInt
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP

		p2 := gc.Prog(mips.ASC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := gc.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)

		gc.Prog(mips.ASYNC)

		p4 := gc.Prog(mips.AADDU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = v.AuxInt
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()

	case ssa.OpMIPSLoweredAtomicAnd,
		ssa.OpMIPSLoweredAtomicOr:
		// SYNC
		// LL	(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		gc.Prog(mips.ASYNC)

		p := gc.Prog(mips.ALL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP

		p1 := gc.Prog(v.Op.Asm())
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = mips.REGTMP
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP

		p2 := gc.Prog(mips.ASC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := gc.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)

		gc.Prog(mips.ASYNC)

	case ssa.OpMIPSLoweredAtomicCas:
		// MOVW $0, Rout
		// SYNC
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVW Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// SYNC
		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		gc.Prog(mips.ASYNC)

		p1 := gc.Prog(mips.ALL)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP

		p2 := gc.Prog(mips.ABNE)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_BRANCH

		p3 := gc.Prog(mips.AMOVW)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg0()

		p4 := gc.Prog(mips.ASC)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_MEM
		p4.To.Reg = v.Args[0].Reg()

		p5 := gc.Prog(mips.ABEQ)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Reg0()
		p5.To.Type = obj.TYPE_BRANCH
		gc.Patch(p5, p1)

		gc.Prog(mips.ASYNC)

		p6 := gc.Prog(obj.ANOP)
		gc.Patch(p2, p6)

	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
	case ssa.OpKeepAlive:
		gc.KeepAlive(v)
	case ssa.OpPhi:
		gc.CheckLoweredPhi(v)
	case ssa.OpMIPSLoweredNilCheck:
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
	case ssa.OpMIPSFPFlagTrue,
		ssa.OpMIPSFPFlagFalse:
		// MOVW		$1, r
		// CMOVF	R0, r

		cmov := mips.ACMOVF
		if v.Op == ssa.OpMIPSFPFlagFalse {
			cmov = mips.ACMOVT
		}
		p := gc.Prog(mips.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p1 := gc.Prog(cmov)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = mips.REGZERO
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg()

	case ssa.OpMIPSLoweredGetClosurePtr:
		// Closure pointer is R22 (mips.REGCTXT).
		gc.CheckLoweredGetClosurePtr(v)
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = map[ssa.BlockKind]struct {
	asm, invasm obj.As
}{
	ssa.BlockMIPSEQ:  {mips.ABEQ, mips.ABNE},
	ssa.BlockMIPSNE:  {mips.ABNE, mips.ABEQ},
	ssa.BlockMIPSLTZ: {mips.ABLTZ, mips.ABGEZ},
	ssa.BlockMIPSGEZ: {mips.ABGEZ, mips.ABLTZ},
	ssa.BlockMIPSLEZ: {mips.ABLEZ, mips.ABGTZ},
	ssa.BlockMIPSGTZ: {mips.ABGTZ, mips.ABLEZ},
	ssa.BlockMIPSFPT: {mips.ABFPT, mips.ABFPF},
	ssa.BlockMIPSFPF: {mips.ABFPF, mips.ABFPT},
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
	case ssa.BlockMIPSEQ, ssa.BlockMIPSNE,
		ssa.BlockMIPSLTZ, ssa.BlockMIPSGEZ,
		ssa.BlockMIPSLEZ, ssa.BlockMIPSGTZ,
		ssa.BlockMIPSFPT, ssa.BlockMIPSFPF:
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
