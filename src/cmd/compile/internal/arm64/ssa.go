// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

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
	"cmd/internal/obj/arm64"
)

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFMOVS
		case 8:
			return arm64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return arm64.AMOVB
			} else {
				return arm64.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return arm64.AMOVH
			} else {
				return arm64.AMOVHU
			}
		case 4:
			if t.IsSigned() {
				return arm64.AMOVW
			} else {
				return arm64.AMOVWU
			}
		case 8:
			return arm64.AMOVD
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFMOVS
		case 8:
			return arm64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return arm64.AMOVB
		case 2:
			return arm64.AMOVH
		case 4:
			return arm64.AMOVW
		case 8:
			return arm64.AMOVD
		}
	}
	panic("bad store type")
}

// loadByType2 returns an opcode that can load consecutive memory locations into 2 registers with type t.
// returns obj.AXXX if no such opcode exists.
func loadByType2(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFLDPS
		case 8:
			return arm64.AFLDPD
		}
	} else {
		switch t.Size() {
		case 4:
			return arm64.ALDPW
		case 8:
			return arm64.ALDP
		}
	}
	return obj.AXXX
}

// storeByType2 returns an opcode that can store registers with type t into 2 consecutive memory locations.
// returns obj.AXXX if no such opcode exists.
func storeByType2(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFSTPS
		case 8:
			return arm64.AFSTPD
		}
	} else {
		switch t.Size() {
		case 4:
			return arm64.ASTPW
		case 8:
			return arm64.ASTP
		}
	}
	return obj.AXXX
}

// makeshift encodes a register shifted by a constant, used as an Offset in Prog.
func makeshift(v *ssa.Value, reg int16, typ int64, s int64) int64 {
	if s < 0 || s >= 64 {
		v.Fatalf("shift out of range: %d", s)
	}
	return int64(reg&31)<<16 | typ | (s&63)<<10
}

// genshift generates a Prog for r = r0 op (r1 shifted by n).
func genshift(s *ssagen.State, v *ssa.Value, as obj.As, r0, r1, r int16, typ int64, n int64) *obj.Prog {
	p := s.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = makeshift(v, r1, typ, n)
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
}

// generate the memory operand for the indexed load/store instructions.
// base and idx are registers.
func genIndexedOperand(op ssa.Op, base, idx int16) obj.Addr {
	// Reg: base register, Index: (shifted) index register
	mop := obj.Addr{Type: obj.TYPE_MEM, Reg: base}
	switch op {
	case ssa.OpARM64MOVDloadidx8, ssa.OpARM64MOVDstoreidx8, ssa.OpARM64MOVDstorezeroidx8,
		ssa.OpARM64FMOVDloadidx8, ssa.OpARM64FMOVDstoreidx8:
		mop.Index = arm64.REG_LSL | 3<<5 | idx&31
	case ssa.OpARM64MOVWloadidx4, ssa.OpARM64MOVWUloadidx4, ssa.OpARM64MOVWstoreidx4, ssa.OpARM64MOVWstorezeroidx4,
		ssa.OpARM64FMOVSloadidx4, ssa.OpARM64FMOVSstoreidx4:
		mop.Index = arm64.REG_LSL | 2<<5 | idx&31
	case ssa.OpARM64MOVHloadidx2, ssa.OpARM64MOVHUloadidx2, ssa.OpARM64MOVHstoreidx2, ssa.OpARM64MOVHstorezeroidx2:
		mop.Index = arm64.REG_LSL | 1<<5 | idx&31
	default: // not shifted
		mop.Index = idx
	}
	return mop
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssa.OpCopy, ssa.OpARM64MOVDreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := arm64.AMOVD
		if v.Type.IsFloat() {
			switch v.Type.Size() {
			case 4:
				as = arm64.AFMOVS
			case 8:
				as = arm64.AFMOVD
			default:
				panic("bad float size")
			}
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssa.OpARM64MOVDnop:
		// nothing to do
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
	case ssa.OpArgIntReg, ssa.OpArgFloatReg:
		ssagen.CheckArgReg(v)
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		args := v.Block.Func.RegArgs
		if len(args) == 0 {
			break
		}
		v.Block.Func.RegArgs = nil // prevent from running again

		for i := 0; i < len(args); i++ {
			a := args[i]
			// Offset by size of the saved LR slot.
			addr := ssagen.SpillSlotAddr(a, arm64.REGSP, base.Ctxt.Arch.FixedFrameSize)
			// Look for double-register operations if we can.
			if i < len(args)-1 {
				b := args[i+1]
				if a.Type.Size() == b.Type.Size() &&
					a.Type.IsFloat() == b.Type.IsFloat() &&
					b.Offset == a.Offset+a.Type.Size() {
					ld := loadByType2(a.Type)
					st := storeByType2(a.Type)
					if ld != obj.AXXX && st != obj.AXXX {
						s.FuncInfo().AddSpill(obj.RegSpill{Reg: a.Reg, Reg2: b.Reg, Addr: addr, Unspill: ld, Spill: st})
						i++ // b is done also, skip it.
						continue
					}
				}
			}
			// Pass the spill/unspill information along to the assembler.
			s.FuncInfo().AddSpill(obj.RegSpill{Reg: a.Reg, Addr: addr, Unspill: loadByType(a.Type), Spill: storeByType(a.Type)})
		}

	case ssa.OpARM64ADD,
		ssa.OpARM64SUB,
		ssa.OpARM64AND,
		ssa.OpARM64OR,
		ssa.OpARM64XOR,
		ssa.OpARM64BIC,
		ssa.OpARM64EON,
		ssa.OpARM64ORN,
		ssa.OpARM64MUL,
		ssa.OpARM64MULW,
		ssa.OpARM64MNEG,
		ssa.OpARM64MNEGW,
		ssa.OpARM64MULH,
		ssa.OpARM64UMULH,
		ssa.OpARM64MULL,
		ssa.OpARM64UMULL,
		ssa.OpARM64DIV,
		ssa.OpARM64UDIV,
		ssa.OpARM64DIVW,
		ssa.OpARM64UDIVW,
		ssa.OpARM64MOD,
		ssa.OpARM64UMOD,
		ssa.OpARM64MODW,
		ssa.OpARM64UMODW,
		ssa.OpARM64SLL,
		ssa.OpARM64SRL,
		ssa.OpARM64SRA,
		ssa.OpARM64FADDS,
		ssa.OpARM64FADDD,
		ssa.OpARM64FSUBS,
		ssa.OpARM64FSUBD,
		ssa.OpARM64FMULS,
		ssa.OpARM64FMULD,
		ssa.OpARM64FNMULS,
		ssa.OpARM64FNMULD,
		ssa.OpARM64FDIVS,
		ssa.OpARM64FDIVD,
		ssa.OpARM64FMINS,
		ssa.OpARM64FMIND,
		ssa.OpARM64FMAXS,
		ssa.OpARM64FMAXD,
		ssa.OpARM64ROR,
		ssa.OpARM64RORW:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARM64FMADDS,
		ssa.OpARM64FMADDD,
		ssa.OpARM64FNMADDS,
		ssa.OpARM64FNMADDD,
		ssa.OpARM64FMSUBS,
		ssa.OpARM64FMSUBD,
		ssa.OpARM64FNMSUBS,
		ssa.OpARM64FNMSUBD,
		ssa.OpARM64MADD,
		ssa.OpARM64MADDW,
		ssa.OpARM64MSUB,
		ssa.OpARM64MSUBW:
		rt := v.Reg()
		ra := v.Args[0].Reg()
		rm := v.Args[1].Reg()
		rn := v.Args[2].Reg()
		p := s.Prog(v.Op.Asm())
		p.Reg = ra
		p.From.Type = obj.TYPE_REG
		p.From.Reg = rm
		p.AddRestSourceReg(rn)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = rt
	case ssa.OpARM64ADDconst,
		ssa.OpARM64SUBconst,
		ssa.OpARM64ANDconst,
		ssa.OpARM64ORconst,
		ssa.OpARM64XORconst,
		ssa.OpARM64SLLconst,
		ssa.OpARM64SRLconst,
		ssa.OpARM64SRAconst,
		ssa.OpARM64RORconst,
		ssa.OpARM64RORWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64ADDSconstflags:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpARM64ADCzerocarry:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64ADCSflags,
		ssa.OpARM64ADDSflags,
		ssa.OpARM64SBCSflags,
		ssa.OpARM64SUBSflags:
		r := v.Reg0()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARM64NEGSflags:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpARM64NGCzerocarry:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64EXTRconst,
		ssa.OpARM64EXTRWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.AddRestSourceReg(v.Args[0].Reg())
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64MVNshiftLL, ssa.OpARM64NEGshiftLL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_LL, v.AuxInt)
	case ssa.OpARM64MVNshiftRL, ssa.OpARM64NEGshiftRL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_LR, v.AuxInt)
	case ssa.OpARM64MVNshiftRA, ssa.OpARM64NEGshiftRA:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_AR, v.AuxInt)
	case ssa.OpARM64MVNshiftRO:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_ROR, v.AuxInt)
	case ssa.OpARM64ADDshiftLL,
		ssa.OpARM64SUBshiftLL,
		ssa.OpARM64ANDshiftLL,
		ssa.OpARM64ORshiftLL,
		ssa.OpARM64XORshiftLL,
		ssa.OpARM64EONshiftLL,
		ssa.OpARM64ORNshiftLL,
		ssa.OpARM64BICshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_LL, v.AuxInt)
	case ssa.OpARM64ADDshiftRL,
		ssa.OpARM64SUBshiftRL,
		ssa.OpARM64ANDshiftRL,
		ssa.OpARM64ORshiftRL,
		ssa.OpARM64XORshiftRL,
		ssa.OpARM64EONshiftRL,
		ssa.OpARM64ORNshiftRL,
		ssa.OpARM64BICshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_LR, v.AuxInt)
	case ssa.OpARM64ADDshiftRA,
		ssa.OpARM64SUBshiftRA,
		ssa.OpARM64ANDshiftRA,
		ssa.OpARM64ORshiftRA,
		ssa.OpARM64XORshiftRA,
		ssa.OpARM64EONshiftRA,
		ssa.OpARM64ORNshiftRA,
		ssa.OpARM64BICshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_AR, v.AuxInt)
	case ssa.OpARM64ANDshiftRO,
		ssa.OpARM64ORshiftRO,
		ssa.OpARM64XORshiftRO,
		ssa.OpARM64EONshiftRO,
		ssa.OpARM64ORNshiftRO,
		ssa.OpARM64BICshiftRO:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_ROR, v.AuxInt)
	case ssa.OpARM64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64FMOVSconst,
		ssa.OpARM64FMOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64FCMPS0,
		ssa.OpARM64FCMPD0:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(0)
		p.Reg = v.Args[0].Reg()
	case ssa.OpARM64CMP,
		ssa.OpARM64CMPW,
		ssa.OpARM64CMN,
		ssa.OpARM64CMNW,
		ssa.OpARM64TST,
		ssa.OpARM64TSTW,
		ssa.OpARM64FCMPS,
		ssa.OpARM64FCMPD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssa.OpARM64CMPconst,
		ssa.OpARM64CMPWconst,
		ssa.OpARM64CMNconst,
		ssa.OpARM64CMNWconst,
		ssa.OpARM64TSTconst,
		ssa.OpARM64TSTWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
	case ssa.OpARM64CMPshiftLL, ssa.OpARM64CMNshiftLL, ssa.OpARM64TSTshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_LL, v.AuxInt)
	case ssa.OpARM64CMPshiftRL, ssa.OpARM64CMNshiftRL, ssa.OpARM64TSTshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_LR, v.AuxInt)
	case ssa.OpARM64CMPshiftRA, ssa.OpARM64CMNshiftRA, ssa.OpARM64TSTshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_AR, v.AuxInt)
	case ssa.OpARM64TSTshiftRO:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_ROR, v.AuxInt)
	case ssa.OpARM64MOVDaddr:
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		var wantreg string
		// MOVD $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R13)
		//               when constant is large, tmp register (R11) may be used
		// - base is SB: load external address from constant pool (use relocation)
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
			// No sym, just MOVD $off(SP), R
			wantreg = "SP"
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
	case ssa.OpARM64MOVBload,
		ssa.OpARM64MOVBUload,
		ssa.OpARM64MOVHload,
		ssa.OpARM64MOVHUload,
		ssa.OpARM64MOVWload,
		ssa.OpARM64MOVWUload,
		ssa.OpARM64MOVDload,
		ssa.OpARM64FMOVSload,
		ssa.OpARM64FMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64LDP, ssa.OpARM64LDPW, ssa.OpARM64LDPSW, ssa.OpARM64FLDPD, ssa.OpARM64FLDPS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg0()
		p.To.Offset = int64(v.Reg1())
	case ssa.OpARM64MOVBloadidx,
		ssa.OpARM64MOVBUloadidx,
		ssa.OpARM64MOVHloadidx,
		ssa.OpARM64MOVHUloadidx,
		ssa.OpARM64MOVWloadidx,
		ssa.OpARM64MOVWUloadidx,
		ssa.OpARM64MOVDloadidx,
		ssa.OpARM64FMOVSloadidx,
		ssa.OpARM64FMOVDloadidx,
		ssa.OpARM64MOVHloadidx2,
		ssa.OpARM64MOVHUloadidx2,
		ssa.OpARM64MOVWloadidx4,
		ssa.OpARM64MOVWUloadidx4,
		ssa.OpARM64MOVDloadidx8,
		ssa.OpARM64FMOVDloadidx8,
		ssa.OpARM64FMOVSloadidx4:
		p := s.Prog(v.Op.Asm())
		p.From = genIndexedOperand(v.Op, v.Args[0].Reg(), v.Args[1].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64LDAR,
		ssa.OpARM64LDARB,
		ssa.OpARM64LDARW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpARM64MOVBstore,
		ssa.OpARM64MOVHstore,
		ssa.OpARM64MOVWstore,
		ssa.OpARM64MOVDstore,
		ssa.OpARM64FMOVSstore,
		ssa.OpARM64FMOVDstore,
		ssa.OpARM64STLRB,
		ssa.OpARM64STLR,
		ssa.OpARM64STLRW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpARM64MOVBstoreidx,
		ssa.OpARM64MOVHstoreidx,
		ssa.OpARM64MOVWstoreidx,
		ssa.OpARM64MOVDstoreidx,
		ssa.OpARM64FMOVSstoreidx,
		ssa.OpARM64FMOVDstoreidx,
		ssa.OpARM64MOVHstoreidx2,
		ssa.OpARM64MOVWstoreidx4,
		ssa.OpARM64FMOVSstoreidx4,
		ssa.OpARM64MOVDstoreidx8,
		ssa.OpARM64FMOVDstoreidx8:
		p := s.Prog(v.Op.Asm())
		p.To = genIndexedOperand(v.Op, v.Args[0].Reg(), v.Args[1].Reg())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
	case ssa.OpARM64STP, ssa.OpARM64STPW, ssa.OpARM64FSTPD, ssa.OpARM64FSTPS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REGREG
		p.From.Reg = v.Args[1].Reg()
		p.From.Offset = int64(v.Args[2].Reg())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpARM64MOVBstorezero,
		ssa.OpARM64MOVHstorezero,
		ssa.OpARM64MOVWstorezero,
		ssa.OpARM64MOVDstorezero:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpARM64MOVBstorezeroidx,
		ssa.OpARM64MOVHstorezeroidx,
		ssa.OpARM64MOVWstorezeroidx,
		ssa.OpARM64MOVDstorezeroidx,
		ssa.OpARM64MOVHstorezeroidx2,
		ssa.OpARM64MOVWstorezeroidx4,
		ssa.OpARM64MOVDstorezeroidx8:
		p := s.Prog(v.Op.Asm())
		p.To = genIndexedOperand(v.Op, v.Args[0].Reg(), v.Args[1].Reg())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
	case ssa.OpARM64MOVQstorezero:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REGREG
		p.From.Reg = arm64.REGZERO
		p.From.Offset = int64(arm64.REGZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpARM64BFI,
		ssa.OpARM64BFXIL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt >> 8
		p.AddRestSourceConst(v.AuxInt & 0xff)
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64SBFIZ,
		ssa.OpARM64SBFX,
		ssa.OpARM64UBFIZ,
		ssa.OpARM64UBFX:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt >> 8
		p.AddRestSourceConst(v.AuxInt & 0xff)
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64LoweredAtomicExchange64,
		ssa.OpARM64LoweredAtomicExchange32,
		ssa.OpARM64LoweredAtomicExchange8:
		// LDAXR	(Rarg0), Rout
		// STLXR	Rarg1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -2(PC)
		var ld, st obj.As
		switch v.Op {
		case ssa.OpARM64LoweredAtomicExchange8:
			ld = arm64.ALDAXRB
			st = arm64.ASTLXRB
		case ssa.OpARM64LoweredAtomicExchange32:
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		case ssa.OpARM64LoweredAtomicExchange64:
			ld = arm64.ALDAXR
			st = arm64.ASTLXR
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(st)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = arm64.REGTMP
		p2 := s.Prog(arm64.ACBNZ)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm64.REGTMP
		p2.To.Type = obj.TYPE_BRANCH
		p2.To.SetTarget(p)
	case ssa.OpARM64LoweredAtomicExchange64Variant,
		ssa.OpARM64LoweredAtomicExchange32Variant,
		ssa.OpARM64LoweredAtomicExchange8Variant:
		var swap obj.As
		switch v.Op {
		case ssa.OpARM64LoweredAtomicExchange8Variant:
			swap = arm64.ASWPALB
		case ssa.OpARM64LoweredAtomicExchange32Variant:
			swap = arm64.ASWPALW
		case ssa.OpARM64LoweredAtomicExchange64Variant:
			swap = arm64.ASWPALD
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// SWPALD	Rarg1, (Rarg0), Rout
		p := s.Prog(swap)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out

	case ssa.OpARM64LoweredAtomicAdd64,
		ssa.OpARM64LoweredAtomicAdd32:
		// LDAXR	(Rarg0), Rout
		// ADD		Rarg1, Rout
		// STLXR	Rout, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		if v.Op == ssa.OpARM64LoweredAtomicAdd32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(arm64.AADD)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out
		p2 := s.Prog(st)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = out
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = r0
		p2.RegTo2 = arm64.REGTMP
		p3 := s.Prog(arm64.ACBNZ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = arm64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
	case ssa.OpARM64LoweredAtomicAdd64Variant,
		ssa.OpARM64LoweredAtomicAdd32Variant:
		// LDADDAL	Rarg1, (Rarg0), Rout
		// ADD		Rarg1, Rout
		op := arm64.ALDADDALD
		if v.Op == ssa.OpARM64LoweredAtomicAdd32Variant {
			op = arm64.ALDADDALW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(op)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out
		p1 := s.Prog(arm64.AADD)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out
	case ssa.OpARM64LoweredAtomicCas64,
		ssa.OpARM64LoweredAtomicCas32:
		// LDAXR	(Rarg0), Rtmp
		// CMP		Rarg1, Rtmp
		// BNE		3(PC)
		// STLXR	Rarg2, (Rarg0), Rtmp
		// CBNZ		Rtmp, -4(PC)
		// CSET		EQ, Rout
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		cmp := arm64.ACMP
		if v.Op == ssa.OpARM64LoweredAtomicCas32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
			cmp = arm64.ACMPW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p1 := s.Prog(cmp)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = arm64.REGTMP
		p2 := s.Prog(arm64.ABNE)
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(st)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = r2
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = r0
		p3.RegTo2 = arm64.REGTMP
		p4 := s.Prog(arm64.ACBNZ)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = arm64.REGTMP
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p)
		p5 := s.Prog(arm64.ACSET)
		p5.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		p5.From.Offset = int64(arm64.SPOP_EQ)
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = out
		p2.To.SetTarget(p5)
	case ssa.OpARM64LoweredAtomicCas64Variant,
		ssa.OpARM64LoweredAtomicCas32Variant:
		// Rarg0: ptr
		// Rarg1: old
		// Rarg2: new
		// MOV  	Rarg1, Rtmp
		// CASAL	Rtmp, (Rarg0), Rarg2
		// CMP  	Rarg1, Rtmp
		// CSET 	EQ, Rout
		cas := arm64.ACASALD
		cmp := arm64.ACMP
		mov := arm64.AMOVD
		if v.Op == ssa.OpARM64LoweredAtomicCas32Variant {
			cas = arm64.ACASALW
			cmp = arm64.ACMPW
			mov = arm64.AMOVW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()

		// MOV  	Rarg1, Rtmp
		p := s.Prog(mov)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP

		// CASAL	Rtmp, (Rarg0), Rarg2
		p1 := s.Prog(cas)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = arm64.REGTMP
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = r2

		// CMP  	Rarg1, Rtmp
		p2 := s.Prog(cmp)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = r1
		p2.Reg = arm64.REGTMP

		// CSET 	EQ, Rout
		p3 := s.Prog(arm64.ACSET)
		p3.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		p3.From.Offset = int64(arm64.SPOP_EQ)
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = out

	case ssa.OpARM64LoweredAtomicAnd64,
		ssa.OpARM64LoweredAtomicOr64,
		ssa.OpARM64LoweredAtomicAnd32,
		ssa.OpARM64LoweredAtomicOr32,
		ssa.OpARM64LoweredAtomicAnd8,
		ssa.OpARM64LoweredAtomicOr8:
		// LDAXR[BW] (Rarg0), Rout
		// AND/OR	Rarg1, Rout, tmp1
		// STLXR[BW] tmp1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		if v.Op == ssa.OpARM64LoweredAtomicAnd32 || v.Op == ssa.OpARM64LoweredAtomicOr32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		}
		if v.Op == ssa.OpARM64LoweredAtomicAnd8 || v.Op == ssa.OpARM64LoweredAtomicOr8 {
			ld = arm64.ALDAXRB
			st = arm64.ASTLXRB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		tmp := v.RegTmp()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(v.Op.Asm())
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = out
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = tmp
		p2 := s.Prog(st)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = tmp
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = r0
		p2.RegTo2 = arm64.REGTMP
		p3 := s.Prog(arm64.ACBNZ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = arm64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)

	case ssa.OpARM64LoweredAtomicAnd8Variant,
		ssa.OpARM64LoweredAtomicAnd32Variant,
		ssa.OpARM64LoweredAtomicAnd64Variant:
		atomic_clear := arm64.ALDCLRALD
		if v.Op == ssa.OpARM64LoweredAtomicAnd32Variant {
			atomic_clear = arm64.ALDCLRALW
		}
		if v.Op == ssa.OpARM64LoweredAtomicAnd8Variant {
			atomic_clear = arm64.ALDCLRALB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// MNV       Rarg1 Rtemp
		p := s.Prog(arm64.AMVN)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP

		// LDCLRAL[BDW]  Rtemp, (Rarg0), Rout
		p1 := s.Prog(atomic_clear)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = arm64.REGTMP
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = out

	case ssa.OpARM64LoweredAtomicOr8Variant,
		ssa.OpARM64LoweredAtomicOr32Variant,
		ssa.OpARM64LoweredAtomicOr64Variant:
		atomic_or := arm64.ALDORALD
		if v.Op == ssa.OpARM64LoweredAtomicOr32Variant {
			atomic_or = arm64.ALDORALW
		}
		if v.Op == ssa.OpARM64LoweredAtomicOr8Variant {
			atomic_or = arm64.ALDORALB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// LDORAL[BDW]  Rarg1, (Rarg0), Rout
		p := s.Prog(atomic_or)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out

	case ssa.OpARM64MOVBreg,
		ssa.OpARM64MOVBUreg,
		ssa.OpARM64MOVHreg,
		ssa.OpARM64MOVHUreg,
		ssa.OpARM64MOVWreg,
		ssa.OpARM64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpARM64MOVDreg {
			a = a.Args[0]
		}
		if a.Op == ssa.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssa.OpARM64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpARM64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpARM64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpARM64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssa.OpARM64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssa.OpARM64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssa.OpARM64MVN,
		ssa.OpARM64NEG,
		ssa.OpARM64FABSD,
		ssa.OpARM64FMOVDfpgp,
		ssa.OpARM64FMOVDgpfp,
		ssa.OpARM64FMOVSfpgp,
		ssa.OpARM64FMOVSgpfp,
		ssa.OpARM64FNEGS,
		ssa.OpARM64FNEGD,
		ssa.OpARM64FSQRTS,
		ssa.OpARM64FSQRTD,
		ssa.OpARM64FCVTZSSW,
		ssa.OpARM64FCVTZSDW,
		ssa.OpARM64FCVTZUSW,
		ssa.OpARM64FCVTZUDW,
		ssa.OpARM64FCVTZSS,
		ssa.OpARM64FCVTZSD,
		ssa.OpARM64FCVTZUS,
		ssa.OpARM64FCVTZUD,
		ssa.OpARM64SCVTFWS,
		ssa.OpARM64SCVTFWD,
		ssa.OpARM64SCVTFS,
		ssa.OpARM64SCVTFD,
		ssa.OpARM64UCVTFWS,
		ssa.OpARM64UCVTFWD,
		ssa.OpARM64UCVTFS,
		ssa.OpARM64UCVTFD,
		ssa.OpARM64FCVTSD,
		ssa.OpARM64FCVTDS,
		ssa.OpARM64REV,
		ssa.OpARM64REVW,
		ssa.OpARM64REV16,
		ssa.OpARM64REV16W,
		ssa.OpARM64RBIT,
		ssa.OpARM64RBITW,
		ssa.OpARM64CLZ,
		ssa.OpARM64CLZW,
		ssa.OpARM64FRINTAD,
		ssa.OpARM64FRINTMD,
		ssa.OpARM64FRINTND,
		ssa.OpARM64FRINTPD,
		ssa.OpARM64FRINTZD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64LoweredRound32F, ssa.OpARM64LoweredRound64F:
		// input is already rounded
	case ssa.OpARM64VCNT:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = (v.Args[0].Reg()-arm64.REG_F0)&31 + arm64.REG_ARNG + ((arm64.ARNG_8B & 15) << 5)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = (v.Reg()-arm64.REG_F0)&31 + arm64.REG_ARNG + ((arm64.ARNG_8B & 15) << 5)
	case ssa.OpARM64VUADDLV:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = (v.Args[0].Reg()-arm64.REG_F0)&31 + arm64.REG_ARNG + ((arm64.ARNG_8B & 15) << 5)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg() - arm64.REG_F0 + arm64.REG_V0
	case ssa.OpARM64CSEL, ssa.OpARM64CSEL0:
		r1 := int16(arm64.REGZERO)
		if v.Op != ssa.OpARM64CSEL0 {
			r1 = v.Args[1].Reg()
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssa.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.Reg = v.Args[0].Reg()
		p.AddRestSourceReg(r1)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64CSINC, ssa.OpARM64CSINV, ssa.OpARM64CSNEG:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssa.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.Reg = v.Args[0].Reg()
		p.AddRestSourceReg(v.Args[1].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64CSETM:
		p := s.Prog(arm64.ACSETM)
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssa.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64DUFFZERO:
		// runtime.duffzero expects start address in R20
		p := s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt
	case ssa.OpARM64LoweredZero:
		// STP.P	(ZR,ZR), 16(R16)
		// CMP	Rarg1, R16
		// BLE	-2(PC)
		// arg1 is the address of the last 16-byte unit to zero
		p := s.Prog(arm64.ASTP)
		p.Scond = arm64.C_XPOST
		p.From.Type = obj.TYPE_REGREG
		p.From.Reg = arm64.REGZERO
		p.From.Offset = int64(arm64.REGZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REG_R16
		p.To.Offset = 16
		p2 := s.Prog(arm64.ACMP)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = arm64.REG_R16
		p3 := s.Prog(arm64.ABLE)
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
	case ssa.OpARM64DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt
	case ssa.OpARM64LoweredMove:
		// LDP.P	16(R16), (R25, Rtmp)
		// STP.P	(R25, Rtmp), 16(R17)
		// CMP	Rarg2, R16
		// BLE	-3(PC)
		// arg2 is the address of the last element of src
		p := s.Prog(arm64.ALDP)
		p.Scond = arm64.C_XPOST
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = arm64.REG_R16
		p.From.Offset = 16
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = arm64.REG_R25
		p.To.Offset = int64(arm64.REGTMP)
		p2 := s.Prog(arm64.ASTP)
		p2.Scond = arm64.C_XPOST
		p2.From.Type = obj.TYPE_REGREG
		p2.From.Reg = arm64.REG_R25
		p2.From.Offset = int64(arm64.REGTMP)
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = arm64.REG_R17
		p2.To.Offset = 16
		p3 := s.Prog(arm64.ACMP)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.Reg = arm64.REG_R16
		p4 := s.Prog(arm64.ABLE)
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p)
	case ssa.OpARM64CALLstatic, ssa.OpARM64CALLclosure, ssa.OpARM64CALLinter:
		s.Call(v)
	case ssa.OpARM64CALLtail:
		s.TailCall(v)
	case ssa.OpARM64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssa.OpARM64LoweredPanicBoundsA, ssa.OpARM64LoweredPanicBoundsB, ssa.OpARM64LoweredPanicBoundsC:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ssagen.BoundsCheckFunc[v.AuxInt]
		s.UseArgs(16) // space used in callee args area by assembly stubs
	case ssa.OpARM64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(arm64.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Line==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssa.OpARM64Equal,
		ssa.OpARM64NotEqual,
		ssa.OpARM64LessThan,
		ssa.OpARM64LessEqual,
		ssa.OpARM64GreaterThan,
		ssa.OpARM64GreaterEqual,
		ssa.OpARM64LessThanU,
		ssa.OpARM64LessEqualU,
		ssa.OpARM64GreaterThanU,
		ssa.OpARM64GreaterEqualU,
		ssa.OpARM64LessThanF,
		ssa.OpARM64LessEqualF,
		ssa.OpARM64GreaterThanF,
		ssa.OpARM64GreaterEqualF,
		ssa.OpARM64NotLessThanF,
		ssa.OpARM64NotLessEqualF,
		ssa.OpARM64NotGreaterThanF,
		ssa.OpARM64NotGreaterEqualF,
		ssa.OpARM64LessThanNoov,
		ssa.OpARM64GreaterEqualNoov:
		// generate boolean values using CSET
		p := s.Prog(arm64.ACSET)
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[v.Op]
		p.From.Offset = int64(condCode)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64PRFM:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.OpARM64LoweredGetClosurePtr:
		// Closure pointer is R26 (arm64.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssa.OpARM64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpARM64DMB:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
	case ssa.OpARM64FlagConstant:
		v.Fatalf("FlagConstant op should never make it to codegen %v", v.LongString())
	case ssa.OpARM64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.OpClobber:
		// MOVW	$0xdeaddead, REGTMP
		// MOVW	REGTMP, (slot)
		// MOVW	REGTMP, 4(slot)
		p := s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0xdeaddead
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p = s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		ssagen.AddAux(&p.To, v)
		p = s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		ssagen.AddAux2(&p.To, v, v.AuxInt+4)
	case ssa.OpClobberReg:
		x := uint64(0xdeaddeaddeaddead)
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(x)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var condBits = map[ssa.Op]arm64.SpecialOperand{
	ssa.OpARM64Equal:         arm64.SPOP_EQ,
	ssa.OpARM64NotEqual:      arm64.SPOP_NE,
	ssa.OpARM64LessThan:      arm64.SPOP_LT,
	ssa.OpARM64LessThanU:     arm64.SPOP_LO,
	ssa.OpARM64LessEqual:     arm64.SPOP_LE,
	ssa.OpARM64LessEqualU:    arm64.SPOP_LS,
	ssa.OpARM64GreaterThan:   arm64.SPOP_GT,
	ssa.OpARM64GreaterThanU:  arm64.SPOP_HI,
	ssa.OpARM64GreaterEqual:  arm64.SPOP_GE,
	ssa.OpARM64GreaterEqualU: arm64.SPOP_HS,
	ssa.OpARM64LessThanF:     arm64.SPOP_MI, // Less than
	ssa.OpARM64LessEqualF:    arm64.SPOP_LS, // Less than or equal to
	ssa.OpARM64GreaterThanF:  arm64.SPOP_GT, // Greater than
	ssa.OpARM64GreaterEqualF: arm64.SPOP_GE, // Greater than or equal to

	// The following condition codes have unordered to handle comparisons related to NaN.
	ssa.OpARM64NotLessThanF:     arm64.SPOP_PL, // Greater than, equal to, or unordered
	ssa.OpARM64NotLessEqualF:    arm64.SPOP_HI, // Greater than or unordered
	ssa.OpARM64NotGreaterThanF:  arm64.SPOP_LE, // Less than, equal to or unordered
	ssa.OpARM64NotGreaterEqualF: arm64.SPOP_LT, // Less than or unordered

	ssa.OpARM64LessThanNoov:     arm64.SPOP_MI, // Less than but without honoring overflow
	ssa.OpARM64GreaterEqualNoov: arm64.SPOP_PL, // Greater than or equal to but without honoring overflow
}

var blockJump = map[ssa.BlockKind]struct {
	asm, invasm obj.As
}{
	ssa.BlockARM64EQ:     {arm64.ABEQ, arm64.ABNE},
	ssa.BlockARM64NE:     {arm64.ABNE, arm64.ABEQ},
	ssa.BlockARM64LT:     {arm64.ABLT, arm64.ABGE},
	ssa.BlockARM64GE:     {arm64.ABGE, arm64.ABLT},
	ssa.BlockARM64LE:     {arm64.ABLE, arm64.ABGT},
	ssa.BlockARM64GT:     {arm64.ABGT, arm64.ABLE},
	ssa.BlockARM64ULT:    {arm64.ABLO, arm64.ABHS},
	ssa.BlockARM64UGE:    {arm64.ABHS, arm64.ABLO},
	ssa.BlockARM64UGT:    {arm64.ABHI, arm64.ABLS},
	ssa.BlockARM64ULE:    {arm64.ABLS, arm64.ABHI},
	ssa.BlockARM64Z:      {arm64.ACBZ, arm64.ACBNZ},
	ssa.BlockARM64NZ:     {arm64.ACBNZ, arm64.ACBZ},
	ssa.BlockARM64ZW:     {arm64.ACBZW, arm64.ACBNZW},
	ssa.BlockARM64NZW:    {arm64.ACBNZW, arm64.ACBZW},
	ssa.BlockARM64TBZ:    {arm64.ATBZ, arm64.ATBNZ},
	ssa.BlockARM64TBNZ:   {arm64.ATBNZ, arm64.ATBZ},
	ssa.BlockARM64FLT:    {arm64.ABMI, arm64.ABPL},
	ssa.BlockARM64FGE:    {arm64.ABGE, arm64.ABLT},
	ssa.BlockARM64FLE:    {arm64.ABLS, arm64.ABHI},
	ssa.BlockARM64FGT:    {arm64.ABGT, arm64.ABLE},
	ssa.BlockARM64LTnoov: {arm64.ABMI, arm64.ABPL},
	ssa.BlockARM64GEnoov: {arm64.ABPL, arm64.ABMI},
}

// To model a 'LEnoov' ('<=' without overflow checking) branching.
var leJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm64.ABEQ, Index: 0}, {Jump: arm64.ABPL, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm64.ABMI, Index: 0}, {Jump: arm64.ABEQ, Index: 0}}, // next == b.Succs[1]
}

// To model a 'GTnoov' ('>' without overflow checking) branching.
var gtJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm64.ABMI, Index: 1}, {Jump: arm64.ABEQ, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm64.ABEQ, Index: 1}, {Jump: arm64.ABPL, Index: 0}}, // next == b.Succs[1]
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

	case ssa.BlockARM64EQ, ssa.BlockARM64NE,
		ssa.BlockARM64LT, ssa.BlockARM64GE,
		ssa.BlockARM64LE, ssa.BlockARM64GT,
		ssa.BlockARM64ULT, ssa.BlockARM64UGT,
		ssa.BlockARM64ULE, ssa.BlockARM64UGE,
		ssa.BlockARM64Z, ssa.BlockARM64NZ,
		ssa.BlockARM64ZW, ssa.BlockARM64NZW,
		ssa.BlockARM64FLT, ssa.BlockARM64FGE,
		ssa.BlockARM64FLE, ssa.BlockARM64FGT,
		ssa.BlockARM64LTnoov, ssa.BlockARM64GEnoov:
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
	case ssa.BlockARM64TBZ, ssa.BlockARM64TBNZ:
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
		p.From.Offset = b.AuxInt
		p.From.Type = obj.TYPE_CONST
		p.Reg = b.Controls[0].Reg()

	case ssa.BlockARM64LEnoov:
		s.CombJump(b, next, &leJumps)
	case ssa.BlockARM64GTnoov:
		s.CombJump(b, next, &gtJumps)

	case ssa.BlockARM64JUMPTABLE:
		// MOVD	(TABLE)(IDX<<3), Rtmp
		// JMP	(Rtmp)
		p := s.Prog(arm64.AMOVD)
		p.From = genIndexedOperand(ssa.OpARM64MOVDloadidx8, b.Controls[1].Reg(), b.Controls[0].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p = s.Prog(obj.AJMP)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGTMP
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
