// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"fmt"
	"math"

	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

// loadByType returns the load instruction of the given type.
func loadByType(t ssa.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm.AMOVF
		case 8:
			return arm.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return arm.AMOVB
			} else {
				return arm.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return arm.AMOVH
			} else {
				return arm.AMOVHU
			}
		case 4:
			return arm.AMOVW
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t ssa.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm.AMOVF
		case 8:
			return arm.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return arm.AMOVB
		case 2:
			return arm.AMOVH
		case 4:
			return arm.AMOVW
		}
	}
	panic("bad store type")
}

// shift type is used as Offset in obj.TYPE_SHIFT operands to encode shifted register operands
type shift int64

// copied from ../../../internal/obj/util.go:/TYPE_SHIFT
func (v shift) String() string {
	op := "<<>>->@>"[((v>>5)&3)<<1:]
	if v&(1<<4) != 0 {
		// register shift
		return fmt.Sprintf("R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15)
	} else {
		// constant shift
		return fmt.Sprintf("R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31)
	}
}

// makeshift encodes a register shifted by a constant
func makeshift(reg int16, typ int64, s int64) shift {
	return shift(int64(reg&0xf) | typ | (s&31)<<7)
}

// genshift generates a Prog for r = r0 op (r1 shifted by s)
func genshift(as obj.As, r0, r1, r int16, typ int64, s int64) *obj.Prog {
	p := gc.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = int64(makeshift(r1, typ, s))
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
}

// makeregshift encodes a register shifted by a register
func makeregshift(r1 int16, typ int64, r2 int16) shift {
	return shift(int64(r1&0xf) | typ | int64(r2&0xf)<<8 | 1<<4)
}

// genregshift generates a Prog for r = r0 op (r1 shifted by r2)
func genregshift(as obj.As, r0, r1, r2, r int16, typ int64) *obj.Prog {
	p := gc.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = int64(makeregshift(r1, typ, r2))
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
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
	case ssa.OpCopy, ssa.OpARMMOVWconvert, ssa.OpARMMOVWreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := arm.AMOVW
		if v.Type.IsFloat() {
			switch v.Type.Size() {
			case 4:
				as = arm.AMOVF
			case 8:
				as = arm.AMOVD
			default:
				panic("bad float size")
			}
		}
		p := gc.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssa.OpARMMOVWnop:
		if v.Reg() != v.Args[0].Reg() {
			v.Fatalf("input[0] and output not in same register %s", v.LongString())
		}
		// nothing to do
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := gc.Prog(loadByType(v.Type))
		gc.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpPhi:
		gc.CheckLoweredPhi(v)
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := gc.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		gc.AddrAuto(&p.To, v)
	case ssa.OpARMUDIVrtcall:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = obj.Linklookup(gc.Ctxt, "udiv", 0)
	case ssa.OpARMADD,
		ssa.OpARMADC,
		ssa.OpARMSUB,
		ssa.OpARMSBC,
		ssa.OpARMRSB,
		ssa.OpARMAND,
		ssa.OpARMOR,
		ssa.OpARMXOR,
		ssa.OpARMBIC,
		ssa.OpARMMUL,
		ssa.OpARMADDF,
		ssa.OpARMADDD,
		ssa.OpARMSUBF,
		ssa.OpARMSUBD,
		ssa.OpARMMULF,
		ssa.OpARMMULD,
		ssa.OpARMDIVF,
		ssa.OpARMDIVD:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMADDS,
		ssa.OpARMSUBS:
		r := v.Reg0()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := gc.Prog(v.Op.Asm())
		p.Scond = arm.C_SBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMSLL,
		ssa.OpARMSRL,
		ssa.OpARMSRA:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMSRAcond:
		// ARM shift instructions uses only the low-order byte of the shift amount
		// generate conditional instructions to deal with large shifts
		// flag is already set
		// SRA.HS	$31, Rarg0, Rdst // shift 31 bits to get the sign bit
		// SRA.LO	Rarg1, Rarg0, Rdst
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := gc.Prog(arm.ASRA)
		p.Scond = arm.C_SCOND_HS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 31
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p = gc.Prog(arm.ASRA)
		p.Scond = arm.C_SCOND_LO
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMADDconst,
		ssa.OpARMADCconst,
		ssa.OpARMSUBconst,
		ssa.OpARMSBCconst,
		ssa.OpARMRSBconst,
		ssa.OpARMRSCconst,
		ssa.OpARMANDconst,
		ssa.OpARMORconst,
		ssa.OpARMXORconst,
		ssa.OpARMBICconst,
		ssa.OpARMSLLconst,
		ssa.OpARMSRLconst,
		ssa.OpARMSRAconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMADDSconst,
		ssa.OpARMSUBSconst,
		ssa.OpARMRSBSconst:
		p := gc.Prog(v.Op.Asm())
		p.Scond = arm.C_SBIT
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpARMSRRconst:
		genshift(arm.AMOVW, 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_RR, v.AuxInt)
	case ssa.OpARMADDshiftLL,
		ssa.OpARMADCshiftLL,
		ssa.OpARMSUBshiftLL,
		ssa.OpARMSBCshiftLL,
		ssa.OpARMRSBshiftLL,
		ssa.OpARMRSCshiftLL,
		ssa.OpARMANDshiftLL,
		ssa.OpARMORshiftLL,
		ssa.OpARMXORshiftLL,
		ssa.OpARMBICshiftLL:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
	case ssa.OpARMADDSshiftLL,
		ssa.OpARMSUBSshiftLL,
		ssa.OpARMRSBSshiftLL:
		p := genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_LL, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssa.OpARMADDshiftRL,
		ssa.OpARMADCshiftRL,
		ssa.OpARMSUBshiftRL,
		ssa.OpARMSBCshiftRL,
		ssa.OpARMRSBshiftRL,
		ssa.OpARMRSCshiftRL,
		ssa.OpARMANDshiftRL,
		ssa.OpARMORshiftRL,
		ssa.OpARMXORshiftRL,
		ssa.OpARMBICshiftRL:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
	case ssa.OpARMADDSshiftRL,
		ssa.OpARMSUBSshiftRL,
		ssa.OpARMRSBSshiftRL:
		p := genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_LR, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssa.OpARMADDshiftRA,
		ssa.OpARMADCshiftRA,
		ssa.OpARMSUBshiftRA,
		ssa.OpARMSBCshiftRA,
		ssa.OpARMRSBshiftRA,
		ssa.OpARMRSCshiftRA,
		ssa.OpARMANDshiftRA,
		ssa.OpARMORshiftRA,
		ssa.OpARMXORshiftRA,
		ssa.OpARMBICshiftRA:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
	case ssa.OpARMADDSshiftRA,
		ssa.OpARMSUBSshiftRA,
		ssa.OpARMRSBSshiftRA:
		p := genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_AR, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssa.OpARMXORshiftRR:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_RR, v.AuxInt)
	case ssa.OpARMMVNshiftLL:
		genshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
	case ssa.OpARMMVNshiftRL:
		genshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
	case ssa.OpARMMVNshiftRA:
		genshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
	case ssa.OpARMMVNshiftLLreg:
		genregshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL)
	case ssa.OpARMMVNshiftRLreg:
		genregshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR)
	case ssa.OpARMMVNshiftRAreg:
		genregshift(v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR)
	case ssa.OpARMADDshiftLLreg,
		ssa.OpARMADCshiftLLreg,
		ssa.OpARMSUBshiftLLreg,
		ssa.OpARMSBCshiftLLreg,
		ssa.OpARMRSBshiftLLreg,
		ssa.OpARMRSCshiftLLreg,
		ssa.OpARMANDshiftLLreg,
		ssa.OpARMORshiftLLreg,
		ssa.OpARMXORshiftLLreg,
		ssa.OpARMBICshiftLLreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_LL)
	case ssa.OpARMADDSshiftLLreg,
		ssa.OpARMSUBSshiftLLreg,
		ssa.OpARMRSBSshiftLLreg:
		p := genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_LL)
		p.Scond = arm.C_SBIT
	case ssa.OpARMADDshiftRLreg,
		ssa.OpARMADCshiftRLreg,
		ssa.OpARMSUBshiftRLreg,
		ssa.OpARMSBCshiftRLreg,
		ssa.OpARMRSBshiftRLreg,
		ssa.OpARMRSCshiftRLreg,
		ssa.OpARMANDshiftRLreg,
		ssa.OpARMORshiftRLreg,
		ssa.OpARMXORshiftRLreg,
		ssa.OpARMBICshiftRLreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_LR)
	case ssa.OpARMADDSshiftRLreg,
		ssa.OpARMSUBSshiftRLreg,
		ssa.OpARMRSBSshiftRLreg:
		p := genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_LR)
		p.Scond = arm.C_SBIT
	case ssa.OpARMADDshiftRAreg,
		ssa.OpARMADCshiftRAreg,
		ssa.OpARMSUBshiftRAreg,
		ssa.OpARMSBCshiftRAreg,
		ssa.OpARMRSBshiftRAreg,
		ssa.OpARMRSCshiftRAreg,
		ssa.OpARMANDshiftRAreg,
		ssa.OpARMORshiftRAreg,
		ssa.OpARMXORshiftRAreg,
		ssa.OpARMBICshiftRAreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_AR)
	case ssa.OpARMADDSshiftRAreg,
		ssa.OpARMSUBSshiftRAreg,
		ssa.OpARMRSBSshiftRAreg:
		p := genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_AR)
		p.Scond = arm.C_SBIT
	case ssa.OpARMHMUL,
		ssa.OpARMHMULU:
		// 32-bit high multiplication
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg()
		p.To.Offset = arm.REGTMP // throw away low 32-bit into tmp register
	case ssa.OpARMMULLU:
		// 32-bit multiplication, results 64-bit, high 32-bit in out0, low 32-bit in out1
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg0()           // high 32-bit
		p.To.Offset = int64(v.Reg1()) // low 32-bit
	case ssa.OpARMMULA:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG2
		p.To.Reg = v.Reg()                   // result
		p.To.Offset = int64(v.Args[2].Reg()) // addend
	case ssa.OpARMMOVWconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMMOVFconst,
		ssa.OpARMMOVDconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMCMP,
		ssa.OpARMCMN,
		ssa.OpARMTST,
		ssa.OpARMTEQ,
		ssa.OpARMCMPF,
		ssa.OpARMCMPD:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		// Special layout in ARM assembly
		// Comparing to x86, the operands of ARM's CMP are reversed.
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssa.OpARMCMPconst,
		ssa.OpARMCMNconst,
		ssa.OpARMTSTconst,
		ssa.OpARMTEQconst:
		// Special layout in ARM assembly
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
	case ssa.OpARMCMPF0,
		ssa.OpARMCMPD0:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
	case ssa.OpARMCMPshiftLL:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_LL, v.AuxInt)
	case ssa.OpARMCMPshiftRL:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_LR, v.AuxInt)
	case ssa.OpARMCMPshiftRA:
		genshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_AR, v.AuxInt)
	case ssa.OpARMCMPshiftLLreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_LL)
	case ssa.OpARMCMPshiftRLreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_LR)
	case ssa.OpARMCMPshiftRAreg:
		genregshift(v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_AR)
	case ssa.OpARMMOVWaddr:
		p := gc.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_ADDR
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		var wantreg string
		// MOVW $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R13)
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
			// No sym, just MOVW $off(SP), R
			wantreg = "SP"
			p.From.Reg = arm.REGSP
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}

	case ssa.OpARMMOVBload,
		ssa.OpARMMOVBUload,
		ssa.OpARMMOVHload,
		ssa.OpARMMOVHUload,
		ssa.OpARMMOVWload,
		ssa.OpARMMOVFload,
		ssa.OpARMMOVDload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMMOVBstore,
		ssa.OpARMMOVHstore,
		ssa.OpARMMOVWstore,
		ssa.OpARMMOVFstore,
		ssa.OpARMMOVDstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		gc.AddAux(&p.To, v)
	case ssa.OpARMMOVWloadidx:
		// this is just shift 0 bits
		fallthrough
	case ssa.OpARMMOVWloadshiftLL:
		p := genshift(v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssa.OpARMMOVWloadshiftRL:
		p := genshift(v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssa.OpARMMOVWloadshiftRA:
		p := genshift(v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssa.OpARMMOVWstoreidx:
		// this is just shift 0 bits
		fallthrough
	case ssa.OpARMMOVWstoreshiftLL:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v.Args[1].Reg(), arm.SHIFT_LL, v.AuxInt))
	case ssa.OpARMMOVWstoreshiftRL:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v.Args[1].Reg(), arm.SHIFT_LR, v.AuxInt))
	case ssa.OpARMMOVWstoreshiftRA:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v.Args[1].Reg(), arm.SHIFT_AR, v.AuxInt))
	case ssa.OpARMMOVBreg,
		ssa.OpARMMOVBUreg,
		ssa.OpARMMOVHreg,
		ssa.OpARMMOVHUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpARMMOVWreg || a.Op == ssa.OpARMMOVWnop {
			a = a.Args[0]
		}
		if a.Op == ssa.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssa.OpARMMOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpARMMOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpARMMOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpARMMOVHUreg && t.Size() == 2 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := gc.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssa.OpARMMVN,
		ssa.OpARMCLZ,
		ssa.OpARMSQRTD,
		ssa.OpARMNEGF,
		ssa.OpARMNEGD,
		ssa.OpARMMOVWF,
		ssa.OpARMMOVWD,
		ssa.OpARMMOVFW,
		ssa.OpARMMOVDW,
		ssa.OpARMMOVFD,
		ssa.OpARMMOVDF:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMMOVWUF,
		ssa.OpARMMOVWUD,
		ssa.OpARMMOVFWU,
		ssa.OpARMMOVDWU:
		p := gc.Prog(v.Op.Asm())
		p.Scond = arm.C_UBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMCMOVWHSconst:
		p := gc.Prog(arm.AMOVW)
		p.Scond = arm.C_SCOND_HS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMCMOVWLSconst:
		p := gc.Prog(arm.AMOVW)
		p.Scond = arm.C_SCOND_LS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARMCALLstatic:
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
	case ssa.OpARMCALLclosure:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpARMCALLdefer:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Deferproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpARMCALLgo:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Newproc.Sym)
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpARMCALLinter:
		p := gc.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = 0
		p.To.Reg = v.Args[0].Reg()
		if gc.Maxarg < v.AuxInt {
			gc.Maxarg = v.AuxInt
		}
	case ssa.OpARMDUFFZERO:
		p := gc.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))
		p.To.Offset = v.AuxInt
	case ssa.OpARMDUFFCOPY:
		p := gc.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))
		p.To.Offset = v.AuxInt
	case ssa.OpARMLoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := gc.Prog(arm.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		if gc.Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			gc.Warnl(v.Line, "generated nil check")
		}
	case ssa.OpARMLoweredZero:
		// MOVW.P	Rarg2, 4(R1)
		// CMP	Rarg1, R1
		// BLE	-2(PC)
		// arg1 is the address of the last element to zero
		// arg2 is known to be zero
		// auxint is alignment
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%4 == 0:
			sz = 4
			mov = arm.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = arm.AMOVH
		default:
			sz = 1
			mov = arm.AMOVB
		}
		p := gc.Prog(mov)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm.REG_R1
		p.To.Offset = sz
		p2 := gc.Prog(arm.ACMP)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = arm.REG_R1
		p3 := gc.Prog(arm.ABLE)
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)
	case ssa.OpARMLoweredMove:
		// MOVW.P	4(R1), Rtmp
		// MOVW.P	Rtmp, 4(R2)
		// CMP	Rarg2, R1
		// BLE	-3(PC)
		// arg2 is the address of the last element of src
		// auxint is alignment
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%4 == 0:
			sz = 4
			mov = arm.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = arm.AMOVH
		default:
			sz = 1
			mov = arm.AMOVB
		}
		p := gc.Prog(mov)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = arm.REG_R1
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		p2 := gc.Prog(mov)
		p2.Scond = arm.C_PBIT
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = arm.REG_R2
		p2.To.Offset = sz
		p3 := gc.Prog(arm.ACMP)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.Reg = arm.REG_R1
		p4 := gc.Prog(arm.ABLE)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)
	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
	case ssa.OpKeepAlive:
		gc.KeepAlive(v)
	case ssa.OpARMEqual,
		ssa.OpARMNotEqual,
		ssa.OpARMLessThan,
		ssa.OpARMLessEqual,
		ssa.OpARMGreaterThan,
		ssa.OpARMGreaterEqual,
		ssa.OpARMLessThanU,
		ssa.OpARMLessEqualU,
		ssa.OpARMGreaterThanU,
		ssa.OpARMGreaterEqualU:
		// generate boolean values
		// use conditional move
		p := gc.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p = gc.Prog(arm.AMOVW)
		p.Scond = condBits[v.Op]
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpSelect0, ssa.OpSelect1:
		// nothing to do
	case ssa.OpARMLoweredGetClosurePtr:
		// Closure pointer is R7 (arm.REGCTXT).
		gc.CheckLoweredGetClosurePtr(v)
	case ssa.OpARMFlagEQ,
		ssa.OpARMFlagLT_ULT,
		ssa.OpARMFlagLT_UGT,
		ssa.OpARMFlagGT_ULT,
		ssa.OpARMFlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssa.OpARMInvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var condBits = map[ssa.Op]uint8{
	ssa.OpARMEqual:         arm.C_SCOND_EQ,
	ssa.OpARMNotEqual:      arm.C_SCOND_NE,
	ssa.OpARMLessThan:      arm.C_SCOND_LT,
	ssa.OpARMLessThanU:     arm.C_SCOND_LO,
	ssa.OpARMLessEqual:     arm.C_SCOND_LE,
	ssa.OpARMLessEqualU:    arm.C_SCOND_LS,
	ssa.OpARMGreaterThan:   arm.C_SCOND_GT,
	ssa.OpARMGreaterThanU:  arm.C_SCOND_HI,
	ssa.OpARMGreaterEqual:  arm.C_SCOND_GE,
	ssa.OpARMGreaterEqualU: arm.C_SCOND_HS,
}

var blockJump = map[ssa.BlockKind]struct {
	asm, invasm obj.As
}{
	ssa.BlockARMEQ:  {arm.ABEQ, arm.ABNE},
	ssa.BlockARMNE:  {arm.ABNE, arm.ABEQ},
	ssa.BlockARMLT:  {arm.ABLT, arm.ABGE},
	ssa.BlockARMGE:  {arm.ABGE, arm.ABLT},
	ssa.BlockARMLE:  {arm.ABLE, arm.ABGT},
	ssa.BlockARMGT:  {arm.ABGT, arm.ABLE},
	ssa.BlockARMULT: {arm.ABLO, arm.ABHS},
	ssa.BlockARMUGE: {arm.ABHS, arm.ABLO},
	ssa.BlockARMUGT: {arm.ABHI, arm.ABLS},
	ssa.BlockARMULE: {arm.ABLS, arm.ABHI},
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
		// defer returns in R0:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := gc.Prog(arm.ACMP)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.Reg = arm.REG_R0
		p = gc.Prog(arm.ABNE)
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

	case ssa.BlockARMEQ, ssa.BlockARMNE,
		ssa.BlockARMLT, ssa.BlockARMGE,
		ssa.BlockARMLE, ssa.BlockARMGT,
		ssa.BlockARMULT, ssa.BlockARMUGT,
		ssa.BlockARMULE, ssa.BlockARMUGE:
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

	default:
		b.Fatalf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
