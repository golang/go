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
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/loong64"
	"internal/abi"
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
	case ssaop.OpCopy, ssaop.OpLOONG64MOVVreg:
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
	case ssaop.OpLOONG64MOVVnop,
		ssaop.OpLOONG64ZERO,
		ssaop.OpLOONG64LoweredRound32F,
		ssaop.OpLOONG64LoweredRound64F:
		// nothing to do
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		r := v.Reg()
		p := s.Prog(loadByType(v.Type, r))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		r := v.Args[0].Reg()
		p := s.Prog(storeByType(v.Type, r))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		ssagen.AddrAuto(&p.To, v)
	case ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
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
	case ssaop.OpLOONG64ADDV,
		ssaop.OpLOONG64SUBV,
		ssaop.OpLOONG64AND,
		ssaop.OpLOONG64OR,
		ssaop.OpLOONG64XOR,
		ssaop.OpLOONG64NOR,
		ssaop.OpLOONG64ANDN,
		ssaop.OpLOONG64ORN,
		ssaop.OpLOONG64SLL,
		ssaop.OpLOONG64SLLV,
		ssaop.OpLOONG64SRL,
		ssaop.OpLOONG64SRLV,
		ssaop.OpLOONG64SRA,
		ssaop.OpLOONG64SRAV,
		ssaop.OpLOONG64ROTR,
		ssaop.OpLOONG64ROTRV,
		ssaop.OpLOONG64ADDF,
		ssaop.OpLOONG64ADDD,
		ssaop.OpLOONG64SUBF,
		ssaop.OpLOONG64SUBD,
		ssaop.OpLOONG64MULF,
		ssaop.OpLOONG64MULD,
		ssaop.OpLOONG64DIVF,
		ssaop.OpLOONG64DIVD,
		ssaop.OpLOONG64MULV, ssaop.OpLOONG64MULHV, ssaop.OpLOONG64MULHVU, ssaop.OpLOONG64MULH, ssaop.OpLOONG64MULHU,
		ssaop.OpLOONG64DIVV, ssaop.OpLOONG64REMV, ssaop.OpLOONG64DIVVU, ssaop.OpLOONG64REMVU,
		ssaop.OpLOONG64MULWVW, ssaop.OpLOONG64MULWVWU,
		ssaop.OpLOONG64FCOPYSGD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64BSTRPICKV,
		ssaop.OpLOONG64BSTRPICKW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		if v.Op == ssaop.OpLOONG64BSTRPICKW {
			p.From.Offset = v.AuxInt >> 5
			p.AddRestSourceConst(v.AuxInt & 0x1f)
		} else {
			p.From.Offset = v.AuxInt >> 6
			p.AddRestSourceConst(v.AuxInt & 0x3f)
		}
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64FMINF,
		ssaop.OpLOONG64FMIND,
		ssaop.OpLOONG64FMAXF,
		ssaop.OpLOONG64FMAXD:
		// ADDD Rarg0, Rarg1, Rout
		// CMPEQD Rarg0, Rarg0, FCC0
		// bceqz FCC0, end
		// CMPEQD Rarg1, Rarg1, FCC0
		// bceqz FCC0, end
		// F(MIN|MAX)(F|D)

		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg()
		add, fcmp := loong64.AADDD, loong64.ACMPEQD
		if v.Op == ssaop.OpLOONG64FMINF || v.Op == ssaop.OpLOONG64FMAXF {
			add = loong64.AADDF
			fcmp = loong64.ACMPEQF
		}
		p1 := s.Prog(add)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r0
		p1.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out

		p2 := s.Prog(fcmp)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = r0
		p2.Reg = r0
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = loong64.REG_FCC0

		p3 := s.Prog(loong64.ABFPF)
		p3.To.Type = obj.TYPE_BRANCH

		p4 := s.Prog(fcmp)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = r1
		p4.Reg = r1
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = loong64.REG_FCC0

		p5 := s.Prog(loong64.ABFPF)
		p5.To.Type = obj.TYPE_BRANCH

		p6 := s.Prog(v.Op.Asm())
		p6.From.Type = obj.TYPE_REG
		p6.From.Reg = r1
		p6.Reg = r0
		p6.To.Type = obj.TYPE_REG
		p6.To.Reg = out

		nop := s.Prog(obj.ANOP)
		p3.To.SetTarget(nop)
		p5.To.SetTarget(nop)

	case ssaop.OpLOONG64SGT,
		ssaop.OpLOONG64SGTU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpLOONG64ADDVconst,
		ssaop.OpLOONG64ADDV16const,
		ssaop.OpLOONG64SUBVconst,
		ssaop.OpLOONG64ANDconst,
		ssaop.OpLOONG64ORconst,
		ssaop.OpLOONG64XORconst,
		ssaop.OpLOONG64SLLconst,
		ssaop.OpLOONG64SLLVconst,
		ssaop.OpLOONG64SRLconst,
		ssaop.OpLOONG64SRLVconst,
		ssaop.OpLOONG64SRAconst,
		ssaop.OpLOONG64SRAVconst,
		ssaop.OpLOONG64ROTRconst,
		ssaop.OpLOONG64ROTRVconst,
		ssaop.OpLOONG64SGTconst,
		ssaop.OpLOONG64SGTUconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64NORconst:
		// MOVV $const, Rtmp
		// NOR  Rtmp, Rarg0, Rout
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REGTMP

		p2 := s.Prog(v.Op.Asm())
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = loong64.REGTMP
		p2.Reg = v.Args[0].Reg()
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg()

	case ssaop.OpLOONG64MOVVconst:
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
	case ssaop.OpLOONG64MOVFconst,
		ssaop.OpLOONG64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpLOONG64CMPEQF,
		ssaop.OpLOONG64CMPEQD,
		ssaop.OpLOONG64CMPGEF,
		ssaop.OpLOONG64CMPGED,
		ssaop.OpLOONG64CMPGTF,
		ssaop.OpLOONG64CMPGTD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REG_FCC0

	case ssaop.OpLOONG64FMADDF,
		ssaop.OpLOONG64FMADDD,
		ssaop.OpLOONG64FMSUBF,
		ssaop.OpLOONG64FMSUBD,
		ssaop.OpLOONG64FNMADDF,
		ssaop.OpLOONG64FNMADDD,
		ssaop.OpLOONG64FNMSUBF,
		ssaop.OpLOONG64FNMSUBD:
		p := s.Prog(v.Op.Asm())
		// r=(FMA x y z) -> FMADDD z, y, x, r
		// the SSA operand order is for taking advantage of
		// commutativity (that only applies for the first two operands)
		r := v.Reg()
		x := v.Args[0].Reg()
		y := v.Args[1].Reg()
		z := v.Args[2].Reg()
		p.From.Type = obj.TYPE_REG
		p.From.Reg = z
		p.Reg = y
		p.AddRestSourceReg(x)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssaop.OpLOONG64MOVVaddr:
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

	case ssaop.OpLOONG64MOVBloadidx,
		ssaop.OpLOONG64MOVBUloadidx,
		ssaop.OpLOONG64MOVHloadidx,
		ssaop.OpLOONG64MOVHUloadidx,
		ssaop.OpLOONG64MOVWloadidx,
		ssaop.OpLOONG64MOVWUloadidx,
		ssaop.OpLOONG64MOVVloadidx,
		ssaop.OpLOONG64MOVFloadidx,
		ssaop.OpLOONG64MOVDloadidx:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_NONE
		p.From.Reg = v.Args[0].Reg()
		p.From.Index = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64MOVBstoreidx,
		ssaop.OpLOONG64MOVHstoreidx,
		ssaop.OpLOONG64MOVWstoreidx,
		ssaop.OpLOONG64MOVVstoreidx,
		ssaop.OpLOONG64MOVFstoreidx,
		ssaop.OpLOONG64MOVDstoreidx:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_NONE
		p.To.Reg = v.Args[0].Reg()
		p.To.Index = v.Args[1].Reg()

	case ssaop.OpLOONG64MOVBload,
		ssaop.OpLOONG64MOVBUload,
		ssaop.OpLOONG64MOVHload,
		ssaop.OpLOONG64MOVHUload,
		ssaop.OpLOONG64MOVWload,
		ssaop.OpLOONG64MOVWUload,
		ssaop.OpLOONG64MOVVload,
		ssaop.OpLOONG64MOVFload,
		ssaop.OpLOONG64MOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpLOONG64MOVBstore,
		ssaop.OpLOONG64MOVHstore,
		ssaop.OpLOONG64MOVWstore,
		ssaop.OpLOONG64MOVVstore,
		ssaop.OpLOONG64MOVFstore,
		ssaop.OpLOONG64MOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpLOONG64MOVBreg,
		ssaop.OpLOONG64MOVBUreg,
		ssaop.OpLOONG64MOVHreg,
		ssaop.OpLOONG64MOVHUreg,
		ssaop.OpLOONG64MOVWreg,
		ssaop.OpLOONG64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssaop.OpCopy || a.Op == ssaop.OpLOONG64MOVVreg {
			a = a.Args[0]
		}
		if a.Op == ssaop.OpLoadReg && loong64.REG_R0 <= a.Reg() && a.Reg() <= loong64.REG_R31 {
			// LoadReg from a narrower type does an extension, except loading
			// to a floating point register. So only eliminate the extension
			// if it is loaded to an integer register.

			t := a.Type
			switch {
			case v.Op == ssaop.OpLOONG64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssaop.OpLOONG64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssaop.OpLOONG64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssaop.OpLOONG64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssaop.OpLOONG64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssaop.OpLOONG64MOVWUreg && t.Size() == 4 && !t.IsSigned():
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

	case ssaop.OpLOONG64MOVWF,
		ssaop.OpLOONG64MOVWD,
		ssaop.OpLOONG64TRUNCFW,
		ssaop.OpLOONG64TRUNCDW,
		ssaop.OpLOONG64MOVVF,
		ssaop.OpLOONG64MOVVD,
		ssaop.OpLOONG64TRUNCFV,
		ssaop.OpLOONG64TRUNCDV,
		ssaop.OpLOONG64MOVFD,
		ssaop.OpLOONG64MOVDF,
		ssaop.OpLOONG64MOVWfpgp,
		ssaop.OpLOONG64MOVWgpfp,
		ssaop.OpLOONG64MOVVfpgp,
		ssaop.OpLOONG64MOVVgpfp,
		ssaop.OpLOONG64NEGF,
		ssaop.OpLOONG64NEGD,
		ssaop.OpLOONG64CLZW,
		ssaop.OpLOONG64CLZV,
		ssaop.OpLOONG64CTZW,
		ssaop.OpLOONG64CTZV,
		ssaop.OpLOONG64SQRTD,
		ssaop.OpLOONG64SQRTF,
		ssaop.OpLOONG64REVB2H,
		ssaop.OpLOONG64REVB2W,
		ssaop.OpLOONG64REVB4H,
		ssaop.OpLOONG64REVBV,
		ssaop.OpLOONG64BITREV4B,
		ssaop.OpLOONG64BITREVW,
		ssaop.OpLOONG64BITREVV,
		ssaop.OpLOONG64ABSF,
		ssaop.OpLOONG64ABSD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64VPCNT64,
		ssaop.OpLOONG64VPCNT32,
		ssaop.OpLOONG64VPCNT16,
		ssaop.OpLOONG64FRINTND,
		ssaop.OpLOONG64FRINTZD,
		ssaop.OpLOONG64FRINTPD,
		ssaop.OpLOONG64FRINTMD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ((v.Args[0].Reg() - loong64.REG_F0) & 31) + loong64.REG_V0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ((v.Reg() - loong64.REG_F0) & 31) + loong64.REG_V0

	case ssaop.OpLOONG64NEGV:
		// SUB from REGZERO
		p := s.Prog(loong64.ASUBVU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64LoweredZero:
		ptrReg := v.Args[0].Reg()
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Zero too small %d", n)
		}

		// Generate Zeroing instructions.
		var off int64
		for n >= 8 {
			// MOVV     ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			off += 8
			n -= 8
		}
		if n != 0 {
			// MOVV     ZR, off+n-8(ptrReg)
			zero8(s, ptrReg, off+n-8)
		}
	case ssaop.OpLOONG64LoweredZeroLoop:
		ptrReg := v.Args[0].Reg()
		endReg := v.RegTmp()
		flagReg := int16(loong64.REGTMP)
		var off int64
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     4 instructions to implement the loop
			//     8 instructions in the loop body
			//   vs
			//     16 instuctions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("ZeroLoop size too small %d", n)
		}

		//    ADDV    n - n%loopSize, ptrReg, endReg
		//    MOVBU   ir.Syms.Loong64HasLSX, flagReg
		//    BNE     flagReg, lsxInit
		// genericLoop:
		//    for off = 0; off < loopSize; off += 8 {
		//            zero8(s, ptrReg, off)
		//    }
		//    ADDV    $loopSize, ptrReg
		//    BNE     endReg, ptrReg, genericLoop
		//    JMP     tail
		// lsxInit:
		//    VXORV   V31, V31, V31
		// lsxLoop:
		//    for off = 0; off < loopSize; off += 16 {
		//            zero16(s, V31, ptrReg, off)
		//    }
		//    ADDV    $loopSize, ptrReg
		//    BNE     endReg, ptrReg, lsxLoop
		// tail:
		//    n %= loopSize
		//    for off = 0; n >= 8; off += 8, n -= 8 {
		//            zero8(s, ptrReg, off)
		//    }
		//
		//    if n != 0 {
		//           zero8(s, ptrReg, off+n-8)
		//    }

		p1 := s.Prog(loong64.AADDV)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = n - n%loopSize
		p1.Reg = ptrReg
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = endReg

		p2 := s.Prog(loong64.AMOVBU)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Name = obj.NAME_EXTERN
		p2.From.Sym = ir.Syms.Loong64HasLSX
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = flagReg

		p3 := s.Prog(loong64.ABNE)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = flagReg
		p3.To.Type = obj.TYPE_BRANCH

		for off = 0; off < loopSize; off += 8 {
			zero8(s, ptrReg, off)
		}

		p4 := s.Prog(loong64.AADDV)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = loopSize
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = ptrReg

		p5 := s.Prog(loong64.ABNE)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = endReg
		p5.Reg = ptrReg
		p5.To.Type = obj.TYPE_BRANCH
		p5.To.SetTarget(p3.Link)

		p6 := s.Prog(obj.AJMP)
		p6.To.Type = obj.TYPE_BRANCH

		p7 := s.Prog(loong64.AVXORV)
		p7.From.Type = obj.TYPE_REG
		p7.From.Reg = loong64.REG_V31
		p7.To.Type = obj.TYPE_REG
		p7.To.Reg = loong64.REG_V31
		p3.To.SetTarget(p7)

		for off = 0; off < loopSize; off += 16 {
			zero16(s, loong64.REG_V31, ptrReg, off)
		}

		p8 := s.Prog(loong64.AADDV)
		p8.From.Type = obj.TYPE_CONST
		p8.From.Offset = loopSize
		p8.To.Type = obj.TYPE_REG
		p8.To.Reg = ptrReg

		p9 := s.Prog(loong64.ABNE)
		p9.From.Type = obj.TYPE_REG
		p9.From.Reg = endReg
		p9.Reg = ptrReg
		p9.To.Type = obj.TYPE_BRANCH
		p9.To.SetTarget(p7.Link)

		p10 := s.Prog(obj.ANOP)
		p6.To.SetTarget(p10)

		// Multiples of the loop size are now done.
		n %= loopSize
		// Write any fractional portion.
		for off = 0; n >= 8; off += 8 {
			// MOVV   ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			n -= 8
		}

		if n != 0 {
			zero8(s, ptrReg, off+n-8)
		}

	case ssaop.OpLOONG64LoweredMove:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		tmpReg := int16(loong64.REG_R23)
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Move too small %d", n)
		}

		var off int64
		for n >= 8 {
			// MOVV     off(srcReg), tmpReg
			// MOVV     tmpReg, off(dstReg)
			move8(s, srcReg, dstReg, tmpReg, off)
			off += 8
			n -= 8
		}

		if n != 0 {
			// MOVV     off+n-8(srcReg), tmpReg
			// MOVV     tmpReg, off+n-8(srcReg)
			move8(s, srcReg, dstReg, tmpReg, off+n-8)
		}
	case ssaop.OpLOONG64LoweredMoveLoop:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		srcEndReg := int16(loong64.REG_R23)
		tmpReg := int16(loong64.REG_R24)
		var off int64
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     4 instructions to implement the loop
			//     8 instructions in the loop body
			//   vs
			//     16 instructions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("MoveLoop size too small %d", n)
		}

		//    ADDV    n - n%loopSize, srcReg, srcEndReg
		// Loop8:
		//    for off = 0; off < loopSize; off += 8 {
		//            move8(s, srcReg, dstReg, tmpReg, off)
		//    }
		//    ADDV    $loopSize, srcReg
		//    ADDV    $loopSize, dstReg
		//    BNE     srcEndReg, srcReg, Loop8
		//
		//    n %= loopSize
		//    for off = 0; n >= 8; off += 8 {
		//           move8(s, srcReg, dstReg, tmpReg, off)
		//           n -= 8
		//    }
		//
		//    if n != 0 {
		//           move8(s, srcReg, dstReg, tmpReg, off+n-8)
		//    }

		p1 := s.Prog(loong64.AADDV)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = n - n%loopSize
		p1.Reg = srcReg
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = srcEndReg

		for off = 0; off < loopSize; off += 8 {
			move8(s, srcReg, dstReg, tmpReg, off)
		}

		p2 := s.Prog(loong64.AADDV)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = loopSize
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = srcReg

		p3 := s.Prog(loong64.AADDV)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = loopSize
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = dstReg

		p4 := s.Prog(loong64.ABNE)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = srcEndReg
		p4.Reg = srcReg
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p1.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		// Copy any fractional portion.
		for off = 0; n >= 8; off += 8 {
			move8(s, srcReg, dstReg, tmpReg, off)
			n -= 8
		}

		if n != 0 {
			move8(s, srcReg, dstReg, tmpReg, off+n-8)
		}

	case ssaop.OpLOONG64CALLstatic, ssaop.OpLOONG64CALLclosure, ssaop.OpLOONG64CALLinter:
		s.Call(v)
	case ssaop.OpLOONG64CALLtail, ssaop.OpLOONG64CALLtailinter:
		s.TailCall(v)
	case ssaop.OpLOONG64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpLOONG64LoweredPubBarrier:
		// DBAR 0x1A
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0x1A

	case ssaop.OpLOONG64LoweredPanicBoundsRR, ssaop.OpLOONG64LoweredPanicBoundsRC, ssaop.OpLOONG64LoweredPanicBoundsCR, ssaop.OpLOONG64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpLOONG64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - loong64.REG_R4)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - loong64.REG_R4)
		case ssaop.OpLOONG64LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - loong64.REG_R4)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(loong64.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = loong64.REG_R4 + int16(yVal)
			}
		case ssaop.OpLOONG64LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - loong64.REG_R4)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(loong64.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = loong64.REG_R4 + int16(xVal)
			}
		case ssaop.OpLOONG64LoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(loong64.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = loong64.REG_R4 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(loong64.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = loong64.REG_R4 + int16(yVal)
			}
		}
		c := abi.BoundsEncode(code, signed, xIsReg, yIsReg, xVal, yVal)

		p := s.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_PanicBounds)
		p.To.SetConst(int64(c))
		p = s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.PanicBounds

	case ssaop.OpLOONG64LoweredAtomicLoad8, ssaop.OpLOONG64LoweredAtomicLoad32, ssaop.OpLOONG64LoweredAtomicLoad64:
		// MOVB	(Rarg0), Rout
		// DBAR	0x14
		as := loong64.AMOVV
		switch v.Op {
		case ssaop.OpLOONG64LoweredAtomicLoad8:
			as = loong64.AMOVBU
		case ssaop.OpLOONG64LoweredAtomicLoad32:
			as = loong64.AMOVWU
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(loong64.ADBAR)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 0x14

	case ssaop.OpLOONG64LoweredAtomicStore8,
		ssaop.OpLOONG64LoweredAtomicStore32,
		ssaop.OpLOONG64LoweredAtomicStore64:
		// DBAR 0x12
		// MOVx (Rarg1), Rout
		// DBAR 0x18
		movx := loong64.AMOVV
		switch v.Op {
		case ssaop.OpLOONG64LoweredAtomicStore8:
			movx = loong64.AMOVB
		case ssaop.OpLOONG64LoweredAtomicStore32:
			movx = loong64.AMOVW
		}
		p := s.Prog(loong64.ADBAR)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0x12

		p1 := s.Prog(movx)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = v.Args[0].Reg()

		p2 := s.Prog(loong64.ADBAR)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = 0x18

	case ssaop.OpLOONG64LoweredAtomicStore8Variant,
		ssaop.OpLOONG64LoweredAtomicStore32Variant,
		ssaop.OpLOONG64LoweredAtomicStore64Variant:
		//AMSWAPx  Rarg1, (Rarg0), Rout
		amswapx := loong64.AAMSWAPDBV
		switch v.Op {
		case ssaop.OpLOONG64LoweredAtomicStore32Variant:
			amswapx = loong64.AAMSWAPDBW
		case ssaop.OpLOONG64LoweredAtomicStore8Variant:
			amswapx = loong64.AAMSWAPDBB
		}
		p := s.Prog(amswapx)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = loong64.REGZERO

	case ssaop.OpLOONG64LoweredAtomicExchange32, ssaop.OpLOONG64LoweredAtomicExchange64:
		// AMSWAPx	Rarg1, (Rarg0), Rout
		amswapx := loong64.AAMSWAPDBV
		if v.Op == ssaop.OpLOONG64LoweredAtomicExchange32 {
			amswapx = loong64.AAMSWAPDBW
		}
		p := s.Prog(amswapx)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssaop.OpLOONG64LoweredAtomicExchange8Variant:
		// AMSWAPDBB	Rarg1, (Rarg0), Rout
		p := s.Prog(loong64.AAMSWAPDBB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssaop.OpLOONG64LoweredAtomicAdd32, ssaop.OpLOONG64LoweredAtomicAdd64:
		// AMADDx  Rarg1, (Rarg0), Rout
		// ADDV    Rarg1, Rout, Rout
		amaddx := loong64.AAMADDDBV
		addx := loong64.AADDV
		if v.Op == ssaop.OpLOONG64LoweredAtomicAdd32 {
			amaddx = loong64.AAMADDDBW
		}
		p := s.Prog(amaddx)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

		p1 := s.Prog(addx)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()

	case ssaop.OpLOONG64LoweredAtomicCas32, ssaop.OpLOONG64LoweredAtomicCas64:
		// MOVV $0, Rout
		// DBAR 0x14
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVV Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// DBAR 0x12
		ll := loong64.ALLV
		sc := loong64.ASCV
		if v.Op == ssaop.OpLOONG64LoweredAtomicCas32 {
			ll = loong64.ALL
			sc = loong64.ASC
		}

		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		p1 := s.Prog(loong64.ADBAR)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 0x14

		p2 := s.Prog(ll)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Reg = v.Args[0].Reg()
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = loong64.REGTMP

		p3 := s.Prog(loong64.ABNE)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[1].Reg()
		p3.Reg = loong64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH

		p4 := s.Prog(loong64.AMOVV)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[2].Reg()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()

		p5 := s.Prog(sc)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Reg0()
		p5.To.Type = obj.TYPE_MEM
		p5.To.Reg = v.Args[0].Reg()

		p6 := s.Prog(loong64.ABEQ)
		p6.From.Type = obj.TYPE_REG
		p6.From.Reg = v.Reg0()
		p6.To.Type = obj.TYPE_BRANCH
		p6.To.SetTarget(p2)

		p7 := s.Prog(loong64.ADBAR)
		p7.From.Type = obj.TYPE_CONST
		p7.From.Offset = 0x12
		p3.To.SetTarget(p7)

	case ssaop.OpLOONG64LoweredAtomicAnd32,
		ssaop.OpLOONG64LoweredAtomicOr32:
		// AM{AND,OR}DBx  Rarg1, (Rarg0), RegZero
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = loong64.REGZERO

	case ssaop.OpLOONG64LoweredAtomicAnd32value,
		ssaop.OpLOONG64LoweredAtomicAnd64value,
		ssaop.OpLOONG64LoweredAtomicOr64value,
		ssaop.OpLOONG64LoweredAtomicOr32value:
		// AM{AND,OR}DBx  Rarg1, (Rarg0), Rout
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssaop.OpLOONG64LoweredAtomicCas64Variant, ssaop.OpLOONG64LoweredAtomicCas32Variant:
		// MOVV         $0, Rout
		// MOVV         Rarg1, Rtmp
		// AMCASDBx     Rarg2, (Rarg0), Rtmp
		// BNE          Rarg1, Rtmp, 2(PC)
		// MOVV         $1, Rout
		// NOP

		amcasx := loong64.AAMCASDBV
		if v.Op == ssaop.OpLOONG64LoweredAtomicCas32Variant {
			amcasx = loong64.AAMCASDBW
		}

		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		p1 := s.Prog(loong64.AMOVV)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = loong64.REGTMP

		p2 := s.Prog(amcasx)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[2].Reg()
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p2.RegTo2 = loong64.REGTMP

		p3 := s.Prog(loong64.ABNE)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[1].Reg()
		p3.Reg = loong64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH

		p4 := s.Prog(loong64.AMOVV)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = 0x1
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()

		p5 := s.Prog(obj.ANOP)
		p3.To.SetTarget(p5)

	case ssaop.OpLOONG64LoweredNilCheck:
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
	case ssaop.OpLOONG64FPFlagTrue,
		ssaop.OpLOONG64FPFlagFalse:
		// MOVV	$0, r
		// BFPF	2(PC)
		// MOVV	$1, r
		branch := loong64.ABFPF
		if v.Op == ssaop.OpLOONG64FPFlagFalse {
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
	case ssaop.OpLOONG64LoweredGetClosurePtr:
		// Closure pointer is R22 (loong64.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpLOONG64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpLOONG64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpLOONG64MASKEQZ, ssaop.OpLOONG64MASKNEZ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpLOONG64PRELD:
		// PRELD (Rarg0), hint
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.AddRestSourceConst(v.AuxInt & 0x1f)

	case ssaop.OpLOONG64PRELDX:
		// PRELDX (Rarg0), $n, $hint
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: (v.AuxInt >> 5) & 0x1fffffffff},
			{Type: obj.TYPE_CONST, Offset: (v.AuxInt >> 0) & 0x1f},
		})

	case ssaop.OpLOONG64ADDshiftLLV:
		// ADDshiftLLV Rarg0, Rarg1, $shift
		// ALSLV $shift, Rarg1, Rarg0, Rtmp
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[1].Reg()
		p.AddRestSourceReg(v.Args[0].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpClobber, ssaop.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = map[block.BlockKind]struct {
	asm, invasm obj.As
}{
	block.BlockLOONG64EQZ:  {loong64.ABEQ, loong64.ABNE},
	block.BlockLOONG64NEZ:  {loong64.ABNE, loong64.ABEQ},
	block.BlockLOONG64LTZ:  {loong64.ABLTZ, loong64.ABGEZ},
	block.BlockLOONG64GEZ:  {loong64.ABGEZ, loong64.ABLTZ},
	block.BlockLOONG64LEZ:  {loong64.ABLEZ, loong64.ABGTZ},
	block.BlockLOONG64GTZ:  {loong64.ABGTZ, loong64.ABLEZ},
	block.BlockLOONG64FPT:  {loong64.ABFPT, loong64.ABFPF},
	block.BlockLOONG64FPF:  {loong64.ABFPF, loong64.ABFPT},
	block.BlockLOONG64BEQ:  {loong64.ABEQ, loong64.ABNE},
	block.BlockLOONG64BNE:  {loong64.ABNE, loong64.ABEQ},
	block.BlockLOONG64BGE:  {loong64.ABGE, loong64.ABLT},
	block.BlockLOONG64BLT:  {loong64.ABLT, loong64.ABGE},
	block.BlockLOONG64BLTU: {loong64.ABLTU, loong64.ABGEU},
	block.BlockLOONG64BGEU: {loong64.ABGEU, loong64.ABLTU},
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	switch b.Kind {
	case block.BlockPlain, block.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case block.BlockExit, block.BlockRetJmp:
	case block.BlockRet:
		s.Prog(obj.ARET)
	case block.BlockLOONG64EQZ, block.BlockLOONG64NEZ,
		block.BlockLOONG64LTZ, block.BlockLOONG64GEZ,
		block.BlockLOONG64LEZ, block.BlockLOONG64GTZ,
		block.BlockLOONG64BEQ, block.BlockLOONG64BNE,
		block.BlockLOONG64BLT, block.BlockLOONG64BGE,
		block.BlockLOONG64BLTU, block.BlockLOONG64BGEU,
		block.BlockLOONG64FPT, block.BlockLOONG64FPF:
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
		switch b.Kind {
		case block.BlockLOONG64BEQ, block.BlockLOONG64BNE,
			block.BlockLOONG64BGE, block.BlockLOONG64BLT,
			block.BlockLOONG64BGEU, block.BlockLOONG64BLTU:
			p.From.Type = obj.TYPE_REG
			p.From.Reg = b.Controls[0].Reg()
			p.Reg = b.Controls[1].Reg()
		case block.BlockLOONG64EQZ, block.BlockLOONG64NEZ,
			block.BlockLOONG64LTZ, block.BlockLOONG64GEZ,
			block.BlockLOONG64LEZ, block.BlockLOONG64GTZ,
			block.BlockLOONG64FPT, block.BlockLOONG64FPF:
			if !b.Controls[0].Type.IsFlags() {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = b.Controls[0].Reg()
			}
		}
	case block.BlockLOONG64JUMPTABLE:
		// ALSLV $3, Rarg0, Rarg1, REGTMP
		// MOVV (REGTMP), REGTMP
		// JMP	(REGTMP)
		p := s.Prog(loong64.AALSLV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 3 // idx*8
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceReg(b.Controls[1].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REGTMP
		p1 := s.Prog(loong64.AMOVV)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = loong64.REGTMP
		p1.From.Offset = 0
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = loong64.REGTMP
		p2 := s.Prog(obj.AJMP)
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = loong64.REGTMP
		// Save jump tables for later resolution of the target blocks.
		s.JumpTables = append(s.JumpTables, b)

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

// move8 copies 8 bytes at src+off to dst+off.
func move8(s *ssagen.State, src, dst, tmp int16, off int64) {
	// MOVV     off(src), tmp
	ld := s.Prog(loong64.AMOVV)
	ld.From.Type = obj.TYPE_MEM
	ld.From.Reg = src
	ld.From.Offset = off
	ld.To.Type = obj.TYPE_REG
	ld.To.Reg = tmp
	// MOVV     tmp, off(dst)
	st := s.Prog(loong64.AMOVV)
	st.From.Type = obj.TYPE_REG
	st.From.Reg = tmp
	st.To.Type = obj.TYPE_MEM
	st.To.Reg = dst
	st.To.Offset = off
}

// zero8 zeroes 8 bytes at reg+off.
func zero8(s *ssagen.State, reg int16, off int64) {
	// MOVV   ZR, off(reg)
	p := s.Prog(loong64.AMOVV)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = loong64.REGZERO
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
}

// zero16 zeroes 16 bytes at reg+off.
func zero16(s *ssagen.State, regZero, regBase int16, off int64) {
	// VMOVQ   regZero, off(regBase)
	p := s.Prog(loong64.AVMOVQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = regZero
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = regBase
	p.To.Offset = off
}
