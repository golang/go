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
	case ssa.OpLOONG64MOVVnop,
		ssa.OpLOONG64ZERO,
		ssa.OpLOONG64LoweredRound32F,
		ssa.OpLOONG64LoweredRound64F:
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
		ssa.OpLOONG64ANDN,
		ssa.OpLOONG64ORN,
		ssa.OpLOONG64SLL,
		ssa.OpLOONG64SLLV,
		ssa.OpLOONG64SRL,
		ssa.OpLOONG64SRLV,
		ssa.OpLOONG64SRA,
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
		ssa.OpLOONG64MULV, ssa.OpLOONG64MULHV, ssa.OpLOONG64MULHVU, ssa.OpLOONG64MULH, ssa.OpLOONG64MULHU,
		ssa.OpLOONG64DIVV, ssa.OpLOONG64REMV, ssa.OpLOONG64DIVVU, ssa.OpLOONG64REMVU,
		ssa.OpLOONG64FCOPYSGD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpLOONG64BSTRPICKV,
		ssa.OpLOONG64BSTRPICKW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		if v.Op == ssa.OpLOONG64BSTRPICKW {
			p.From.Offset = v.AuxInt >> 5
			p.AddRestSourceConst(v.AuxInt & 0x1f)
		} else {
			p.From.Offset = v.AuxInt >> 6
			p.AddRestSourceConst(v.AuxInt & 0x3f)
		}
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpLOONG64FMINF,
		ssa.OpLOONG64FMIND,
		ssa.OpLOONG64FMAXF,
		ssa.OpLOONG64FMAXD:
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
		if v.Op == ssa.OpLOONG64FMINF || v.Op == ssa.OpLOONG64FMAXF {
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

	case ssa.OpLOONG64SGT,
		ssa.OpLOONG64SGTU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpLOONG64ADDVconst,
		ssa.OpLOONG64ADDV16const,
		ssa.OpLOONG64SUBVconst,
		ssa.OpLOONG64ANDconst,
		ssa.OpLOONG64ORconst,
		ssa.OpLOONG64XORconst,
		ssa.OpLOONG64SLLconst,
		ssa.OpLOONG64SLLVconst,
		ssa.OpLOONG64SRLconst,
		ssa.OpLOONG64SRLVconst,
		ssa.OpLOONG64SRAconst,
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

	case ssa.OpLOONG64NORconst:
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
		p.To.Type = obj.TYPE_REG
		p.To.Reg = loong64.REG_FCC0

	case ssa.OpLOONG64FMADDF,
		ssa.OpLOONG64FMADDD,
		ssa.OpLOONG64FMSUBF,
		ssa.OpLOONG64FMSUBD,
		ssa.OpLOONG64FNMADDF,
		ssa.OpLOONG64FNMADDD,
		ssa.OpLOONG64FNMSUBF,
		ssa.OpLOONG64FNMSUBD:
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

	case ssa.OpLOONG64MOVBloadidx,
		ssa.OpLOONG64MOVBUloadidx,
		ssa.OpLOONG64MOVHloadidx,
		ssa.OpLOONG64MOVHUloadidx,
		ssa.OpLOONG64MOVWloadidx,
		ssa.OpLOONG64MOVWUloadidx,
		ssa.OpLOONG64MOVVloadidx,
		ssa.OpLOONG64MOVFloadidx,
		ssa.OpLOONG64MOVDloadidx:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Name = obj.NAME_NONE
		p.From.Reg = v.Args[0].Reg()
		p.From.Index = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpLOONG64MOVBstoreidx,
		ssa.OpLOONG64MOVHstoreidx,
		ssa.OpLOONG64MOVWstoreidx,
		ssa.OpLOONG64MOVVstoreidx,
		ssa.OpLOONG64MOVFstoreidx,
		ssa.OpLOONG64MOVDstoreidx:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_NONE
		p.To.Reg = v.Args[0].Reg()
		p.To.Index = v.Args[1].Reg()

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
		ssa.OpLOONG64MOVWfpgp,
		ssa.OpLOONG64MOVWgpfp,
		ssa.OpLOONG64MOVVfpgp,
		ssa.OpLOONG64MOVVgpfp,
		ssa.OpLOONG64NEGF,
		ssa.OpLOONG64NEGD,
		ssa.OpLOONG64CLZW,
		ssa.OpLOONG64CLZV,
		ssa.OpLOONG64CTZW,
		ssa.OpLOONG64CTZV,
		ssa.OpLOONG64SQRTD,
		ssa.OpLOONG64SQRTF,
		ssa.OpLOONG64REVB2H,
		ssa.OpLOONG64REVB2W,
		ssa.OpLOONG64REVB4H,
		ssa.OpLOONG64REVBV,
		ssa.OpLOONG64BITREV4B,
		ssa.OpLOONG64BITREVW,
		ssa.OpLOONG64BITREVV,
		ssa.OpLOONG64ABSD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpLOONG64VPCNT64,
		ssa.OpLOONG64VPCNT32,
		ssa.OpLOONG64VPCNT16:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = ((v.Args[0].Reg() - loong64.REG_F0) & 31) + loong64.REG_V0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ((v.Reg() - loong64.REG_F0) & 31) + loong64.REG_V0

	case ssa.OpLOONG64NEGV:
		// SUB from REGZERO
		p := s.Prog(loong64.ASUBVU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = loong64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpLOONG64LoweredZero:
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
	case ssa.OpLOONG64LoweredZeroLoop:
		ptrReg := v.Args[0].Reg()
		countReg := v.RegTmp()
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
			v.Fatalf("ZeroLoop size tool small %d", n)
		}

		// Put iteration count in a register.
		//   MOVV     $n/loopSize, countReg
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Zero loopSize bytes starting at ptrReg.
		for range loopSize / 8 {
			// MOVV     ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			off += 8
		}

		// Increment ptrReg by loopSize.
		//   ADDV     $loopSize, ptrReg
		p = s.Prog(loong64.AADDV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = ptrReg

		// Decrement loop count.
		//   SUBV     $1, countReg
		p = s.Prog(loong64.ASUBV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg

		// Jump to loop header if we're not done yet.
		//   BNE     countReg, loop header
		p = s.Prog(loong64.ABNE)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = countReg
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		off = 0
		// Write any fractional portion.
		for n >= 8 {
			// MOVV     ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			off += 8
			n -= 8
		}

		if n != 0 {
			zero8(s, ptrReg, off+n-8)
		}

	case ssa.OpLOONG64LoweredMove:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		tmpReg := int16(loong64.REG_R20)
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
	case ssa.OpLOONG64LoweredMoveLoop:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		countReg := int16(loong64.REG_R20)
		tmpReg := int16(loong64.REG_R21)
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
			v.Fatalf("ZeroLoop size too small %d", n)
		}

		// Put iteration count in a register.
		//   MOVV     $n/loopSize, countReg
		p := s.Prog(loong64.AMOVV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Move loopSize bytes starting at srcReg to dstReg.
		for range loopSize / 8 {
			// MOVV     off(srcReg), tmpReg
			// MOVV     tmpReg, off(dstReg)
			move8(s, srcReg, dstReg, tmpReg, off)
			off += 8
		}

		// Increment srcReg and destReg by loopSize.
		//   ADDV     $loopSize, srcReg
		p = s.Prog(loong64.AADDV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = srcReg
		//   ADDV     $loopSize, dstReg
		p = s.Prog(loong64.AADDV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = dstReg

		// Decrement loop count.
		//   SUBV     $1, countReg
		p = s.Prog(loong64.ASUBV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg

		// Jump to loop header if we're not done yet.
		//   BNE     countReg, loop header
		p = s.Prog(loong64.ABNE)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = countReg
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		off = 0
		// Copy any fractional portion.
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

	case ssa.OpLOONG64LoweredPubBarrier:
		// DBAR 0x1A
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0x1A

	case ssa.OpLOONG64LoweredPanicBoundsRR, ssa.OpLOONG64LoweredPanicBoundsRC, ssa.OpLOONG64LoweredPanicBoundsCR, ssa.OpLOONG64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssa.OpLOONG64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - loong64.REG_R4)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - loong64.REG_R4)
		case ssa.OpLOONG64LoweredPanicBoundsRC:
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
		case ssa.OpLOONG64LoweredPanicBoundsCR:
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
		case ssa.OpLOONG64LoweredPanicBoundsCC:
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

	case ssa.OpLOONG64LoweredAtomicLoad8, ssa.OpLOONG64LoweredAtomicLoad32, ssa.OpLOONG64LoweredAtomicLoad64:
		// MOVB	(Rarg0), Rout
		// DBAR	0x14
		as := loong64.AMOVV
		switch v.Op {
		case ssa.OpLOONG64LoweredAtomicLoad8:
			as = loong64.AMOVB
		case ssa.OpLOONG64LoweredAtomicLoad32:
			as = loong64.AMOVW
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(loong64.ADBAR)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = 0x14

	case ssa.OpLOONG64LoweredAtomicStore8,
		ssa.OpLOONG64LoweredAtomicStore32,
		ssa.OpLOONG64LoweredAtomicStore64:
		// DBAR 0x12
		// MOVx (Rarg1), Rout
		// DBAR 0x18
		movx := loong64.AMOVV
		switch v.Op {
		case ssa.OpLOONG64LoweredAtomicStore8:
			movx = loong64.AMOVB
		case ssa.OpLOONG64LoweredAtomicStore32:
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

	case ssa.OpLOONG64LoweredAtomicStore8Variant,
		ssa.OpLOONG64LoweredAtomicStore32Variant,
		ssa.OpLOONG64LoweredAtomicStore64Variant:
		//AMSWAPx  Rarg1, (Rarg0), Rout
		amswapx := loong64.AAMSWAPDBV
		switch v.Op {
		case ssa.OpLOONG64LoweredAtomicStore32Variant:
			amswapx = loong64.AAMSWAPDBW
		case ssa.OpLOONG64LoweredAtomicStore8Variant:
			amswapx = loong64.AAMSWAPDBB
		}
		p := s.Prog(amswapx)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = loong64.REGZERO

	case ssa.OpLOONG64LoweredAtomicExchange32, ssa.OpLOONG64LoweredAtomicExchange64:
		// AMSWAPx	Rarg1, (Rarg0), Rout
		amswapx := loong64.AAMSWAPDBV
		if v.Op == ssa.OpLOONG64LoweredAtomicExchange32 {
			amswapx = loong64.AAMSWAPDBW
		}
		p := s.Prog(amswapx)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssa.OpLOONG64LoweredAtomicExchange8Variant:
		// AMSWAPDBB	Rarg1, (Rarg0), Rout
		p := s.Prog(loong64.AAMSWAPDBB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssa.OpLOONG64LoweredAtomicAdd32, ssa.OpLOONG64LoweredAtomicAdd64:
		// AMADDx  Rarg1, (Rarg0), Rout
		// ADDV    Rarg1, Rout, Rout
		amaddx := loong64.AAMADDDBV
		addx := loong64.AADDV
		if v.Op == ssa.OpLOONG64LoweredAtomicAdd32 {
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

	case ssa.OpLOONG64LoweredAtomicCas32, ssa.OpLOONG64LoweredAtomicCas64:
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
		if v.Op == ssa.OpLOONG64LoweredAtomicCas32 {
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

	case ssa.OpLOONG64LoweredAtomicAnd32,
		ssa.OpLOONG64LoweredAtomicOr32:
		// AM{AND,OR}DBx  Rarg1, (Rarg0), RegZero
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = loong64.REGZERO

	case ssa.OpLOONG64LoweredAtomicAnd32value,
		ssa.OpLOONG64LoweredAtomicAnd64value,
		ssa.OpLOONG64LoweredAtomicOr64value,
		ssa.OpLOONG64LoweredAtomicOr32value:
		// AM{AND,OR}DBx  Rarg1, (Rarg0), Rout
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssa.OpLOONG64LoweredAtomicCas64Variant, ssa.OpLOONG64LoweredAtomicCas32Variant:
		// MOVV         $0, Rout
		// MOVV         Rarg1, Rtmp
		// AMCASDBx     Rarg2, (Rarg0), Rtmp
		// BNE          Rarg1, Rtmp, 2(PC)
		// MOVV         $1, Rout
		// NOP

		amcasx := loong64.AAMCASDBV
		if v.Op == ssa.OpLOONG64LoweredAtomicCas32Variant {
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

	case ssa.OpLOONG64PRELD:
		// PRELD (Rarg0), hint
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.AddRestSourceConst(v.AuxInt & 0x1f)

	case ssa.OpLOONG64PRELDX:
		// PRELDX (Rarg0), $n, $hint
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: (v.AuxInt >> 5) & 0x1fffffffff},
			{Type: obj.TYPE_CONST, Offset: (v.AuxInt >> 0) & 0x1f},
		})

	case ssa.OpLOONG64ADDshiftLLV:
		// ADDshiftLLV Rarg0, Rarg1, $shift
		// ALSLV $shift, Rarg1, Rarg0, Rtmp
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[1].Reg()
		p.AddRestSourceReg(v.Args[0].Reg())
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
	ssa.BlockLOONG64EQZ:  {loong64.ABEQ, loong64.ABNE},
	ssa.BlockLOONG64NEZ:  {loong64.ABNE, loong64.ABEQ},
	ssa.BlockLOONG64LTZ:  {loong64.ABLTZ, loong64.ABGEZ},
	ssa.BlockLOONG64GEZ:  {loong64.ABGEZ, loong64.ABLTZ},
	ssa.BlockLOONG64LEZ:  {loong64.ABLEZ, loong64.ABGTZ},
	ssa.BlockLOONG64GTZ:  {loong64.ABGTZ, loong64.ABLEZ},
	ssa.BlockLOONG64FPT:  {loong64.ABFPT, loong64.ABFPF},
	ssa.BlockLOONG64FPF:  {loong64.ABFPF, loong64.ABFPT},
	ssa.BlockLOONG64BEQ:  {loong64.ABEQ, loong64.ABNE},
	ssa.BlockLOONG64BNE:  {loong64.ABNE, loong64.ABEQ},
	ssa.BlockLOONG64BGE:  {loong64.ABGE, loong64.ABLT},
	ssa.BlockLOONG64BLT:  {loong64.ABLT, loong64.ABGE},
	ssa.BlockLOONG64BLTU: {loong64.ABLTU, loong64.ABGEU},
	ssa.BlockLOONG64BGEU: {loong64.ABGEU, loong64.ABLTU},
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
	case ssa.BlockLOONG64EQZ, ssa.BlockLOONG64NEZ,
		ssa.BlockLOONG64LTZ, ssa.BlockLOONG64GEZ,
		ssa.BlockLOONG64LEZ, ssa.BlockLOONG64GTZ,
		ssa.BlockLOONG64BEQ, ssa.BlockLOONG64BNE,
		ssa.BlockLOONG64BLT, ssa.BlockLOONG64BGE,
		ssa.BlockLOONG64BLTU, ssa.BlockLOONG64BGEU,
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
		switch b.Kind {
		case ssa.BlockLOONG64BEQ, ssa.BlockLOONG64BNE,
			ssa.BlockLOONG64BGE, ssa.BlockLOONG64BLT,
			ssa.BlockLOONG64BGEU, ssa.BlockLOONG64BLTU:
			p.From.Type = obj.TYPE_REG
			p.From.Reg = b.Controls[0].Reg()
			p.Reg = b.Controls[1].Reg()
		case ssa.BlockLOONG64EQZ, ssa.BlockLOONG64NEZ,
			ssa.BlockLOONG64LTZ, ssa.BlockLOONG64GEZ,
			ssa.BlockLOONG64LEZ, ssa.BlockLOONG64GTZ,
			ssa.BlockLOONG64FPT, ssa.BlockLOONG64FPF:
			if !b.Controls[0].Type.IsFlags() {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = b.Controls[0].Reg()
			}
		}
	case ssa.BlockLOONG64JUMPTABLE:
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
	// MOVV     ZR, off(reg)
	p := s.Prog(loong64.AMOVV)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = loong64.REGZERO
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
}
