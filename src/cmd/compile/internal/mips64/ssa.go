// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
	"internal/abi"
)

// isFPreg reports whether r is an FP register.
func isFPreg(r int16) bool {
	return mips.REG_F0 <= r && r <= mips.REG_F31
}

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type, r int16) obj.As {
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
func storeByType(t *types.Type, r int16) obj.As {
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

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpCopy, ssaop.OpMIPS64MOVVreg:
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
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssaop.OpMIPS64MOVVnop, ssaop.OpMIPS64ZERO:
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
	case ssaop.OpMIPS64ADDV,
		ssaop.OpMIPS64SUBV,
		ssaop.OpMIPS64AND,
		ssaop.OpMIPS64OR,
		ssaop.OpMIPS64XOR,
		ssaop.OpMIPS64NOR,
		ssaop.OpMIPS64SLLV,
		ssaop.OpMIPS64SRLV,
		ssaop.OpMIPS64SRAV,
		ssaop.OpMIPS64ADDF,
		ssaop.OpMIPS64ADDD,
		ssaop.OpMIPS64SUBF,
		ssaop.OpMIPS64SUBD,
		ssaop.OpMIPS64MULF,
		ssaop.OpMIPS64MULD,
		ssaop.OpMIPS64DIVF,
		ssaop.OpMIPS64DIVD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64SGT,
		ssaop.OpMIPS64SGTU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64ADDVconst,
		ssaop.OpMIPS64SUBVconst,
		ssaop.OpMIPS64ANDconst,
		ssaop.OpMIPS64ORconst,
		ssaop.OpMIPS64XORconst,
		ssaop.OpMIPS64SLLVconst,
		ssaop.OpMIPS64SRLVconst,
		ssaop.OpMIPS64SRAVconst,
		ssaop.OpMIPS64SGTconst,
		ssaop.OpMIPS64SGTUconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64MULV,
		ssaop.OpMIPS64MULVU,
		ssaop.OpMIPS64DIVV,
		ssaop.OpMIPS64DIVVU:
		// HI, LO results exist in low quality registers that can't
		// be stored without using REGTMP.
		// This used to cause corruptions due to nested REGTMP issues
		// since store might also need REGTMP to materialize the address.
		// Instead we move the result into high quality registers.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
		p1 := s.Prog(mips.AMOVV)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = mips.REG_HI
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()
		p2 := s.Prog(mips.AMOVV)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REG_LO
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg1()
	case ssaop.OpMIPS64MOVVconst:
		r := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if isFPreg(r) {
			// cannot move into FP registers, use TMP as intermediate
			p.To.Reg = mips.REGTMP
			p = s.Prog(mips.AMOVV)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = mips.REGTMP
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssaop.OpMIPS64MOVFconst,
		ssaop.OpMIPS64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64CMPEQF,
		ssaop.OpMIPS64CMPEQD,
		ssaop.OpMIPS64CMPGEF,
		ssaop.OpMIPS64CMPGED,
		ssaop.OpMIPS64CMPGTF,
		ssaop.OpMIPS64CMPGTD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
	case ssaop.OpMIPS64MOVVaddr:
		p := s.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		var wantreg string
		// MOVV $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R29)
		//               when constant is large, tmp register (R23) may be used
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
	case ssaop.OpMIPS64MOVBload,
		ssaop.OpMIPS64MOVBUload,
		ssaop.OpMIPS64MOVHload,
		ssaop.OpMIPS64MOVHUload,
		ssaop.OpMIPS64MOVWload,
		ssaop.OpMIPS64MOVWUload,
		ssaop.OpMIPS64MOVVload,
		ssaop.OpMIPS64MOVFload,
		ssaop.OpMIPS64MOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64MOVBstore,
		ssaop.OpMIPS64MOVHstore,
		ssaop.OpMIPS64MOVWstore,
		ssaop.OpMIPS64MOVVstore,
		ssaop.OpMIPS64MOVFstore,
		ssaop.OpMIPS64MOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpMIPS64MOVBreg,
		ssaop.OpMIPS64MOVBUreg,
		ssaop.OpMIPS64MOVHreg,
		ssaop.OpMIPS64MOVHUreg,
		ssaop.OpMIPS64MOVWreg,
		ssaop.OpMIPS64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssaop.OpCopy || a.Op == ssaop.OpMIPS64MOVVreg {
			a = a.Args[0]
		}
		if a.Op == ssaop.OpLoadReg && mips.REG_R0 <= a.Reg() && a.Reg() <= mips.REG_R31 {
			// LoadReg from a narrower type does an extension, except loading
			// to a floating point register. So only eliminate the extension
			// if it is loaded to an integer register.
			t := a.Type
			switch {
			case v.Op == ssaop.OpMIPS64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssaop.OpMIPS64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssaop.OpMIPS64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssaop.OpMIPS64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssaop.OpMIPS64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssaop.OpMIPS64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := s.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssaop.OpMIPS64MOVWF,
		ssaop.OpMIPS64MOVWD,
		ssaop.OpMIPS64TRUNCFW,
		ssaop.OpMIPS64TRUNCDW,
		ssaop.OpMIPS64MOVVF,
		ssaop.OpMIPS64MOVVD,
		ssaop.OpMIPS64TRUNCFV,
		ssaop.OpMIPS64TRUNCDV,
		ssaop.OpMIPS64MOVFD,
		ssaop.OpMIPS64MOVDF,
		ssaop.OpMIPS64MOVWfpgp,
		ssaop.OpMIPS64MOVWgpfp,
		ssaop.OpMIPS64MOVVfpgp,
		ssaop.OpMIPS64MOVVgpfp,
		ssaop.OpMIPS64NEGF,
		ssaop.OpMIPS64NEGD,
		ssaop.OpMIPS64ABSF,
		ssaop.OpMIPS64ABSD,
		ssaop.OpMIPS64SQRTF,
		ssaop.OpMIPS64SQRTD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64NEGV:
		// SUB from REGZERO
		p := s.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64DUFFZERO:
		// runtime.duffzero expects start address - 8 in R1
		p := s.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 8
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p = s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt
	case ssaop.OpMIPS64LoweredZero:
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
		p := s.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p2 := s.Prog(mov)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGZERO
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = mips.REG_R1
		p2.To.Offset = sz
		p3 := s.Prog(mips.AADDVU)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = sz
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = mips.REG_R1
		p4 := s.Prog(mips.ABNE)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[1].Reg()
		p4.Reg = mips.REG_R1
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p2)
	case ssaop.OpMIPS64DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt
	case ssaop.OpMIPS64LoweredMove:
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
		p := s.Prog(mips.ASUBVU)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REG_R1
		p2 := s.Prog(mov)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Reg = mips.REG_R1
		p2.From.Offset = sz
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = mips.REGTMP
		p3 := s.Prog(mov)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = mips.REG_R2
		p4 := s.Prog(mips.AADDVU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = sz
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = mips.REG_R1
		p5 := s.Prog(mips.AADDVU)
		p5.From.Type = obj.TYPE_CONST
		p5.From.Offset = sz
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = mips.REG_R2
		p6 := s.Prog(mips.ABNE)
		p6.From.Type = obj.TYPE_REG
		p6.From.Reg = v.Args[2].Reg()
		p6.Reg = mips.REG_R1
		p6.To.Type = obj.TYPE_BRANCH
		p6.To.SetTarget(p2)
	case ssaop.OpMIPS64CALLstatic, ssaop.OpMIPS64CALLclosure, ssaop.OpMIPS64CALLinter:
		s.Call(v)
	case ssaop.OpMIPS64CALLtail, ssaop.OpMIPS64CALLtailinter:
		s.TailCall(v)
	case ssaop.OpMIPS64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpMIPS64LoweredPanicBoundsRR, ssaop.OpMIPS64LoweredPanicBoundsRC, ssaop.OpMIPS64LoweredPanicBoundsCR, ssaop.OpMIPS64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpMIPS64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - mips.REG_R1)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - mips.REG_R1)
		case ssaop.OpMIPS64LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - mips.REG_R1)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = mips.REG_R1 + int16(yVal)
			}
		case ssaop.OpMIPS64LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - mips.REG_R1)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = mips.REG_R1 + int16(xVal)
			}
		case ssaop.OpMIPS64LoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = mips.REG_R1 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(mips.AMOVV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = mips.REG_R1 + int16(yVal)
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

	case ssaop.OpMIPS64LoweredAtomicLoad8, ssaop.OpMIPS64LoweredAtomicLoad32, ssaop.OpMIPS64LoweredAtomicLoad64:
		as := mips.AMOVV
		switch v.Op {
		case ssaop.OpMIPS64LoweredAtomicLoad8:
			as = mips.AMOVB
		case ssaop.OpMIPS64LoweredAtomicLoad32:
			as = mips.AMOVW
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		s.Prog(mips.ASYNC)
	case ssaop.OpMIPS64LoweredAtomicStore8, ssaop.OpMIPS64LoweredAtomicStore32, ssaop.OpMIPS64LoweredAtomicStore64:
		as := mips.AMOVV
		switch v.Op {
		case ssaop.OpMIPS64LoweredAtomicStore8:
			as = mips.AMOVB
		case ssaop.OpMIPS64LoweredAtomicStore32:
			as = mips.AMOVW
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		s.Prog(mips.ASYNC)
	case ssaop.OpMIPS64LoweredAtomicStorezero32, ssaop.OpMIPS64LoweredAtomicStorezero64:
		as := mips.AMOVV
		if v.Op == ssaop.OpMIPS64LoweredAtomicStorezero32 {
			as = mips.AMOVW
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		s.Prog(mips.ASYNC)
	case ssaop.OpMIPS64LoweredAtomicExchange32, ssaop.OpMIPS64LoweredAtomicExchange64:
		// SYNC
		// MOVV	Rarg1, Rtmp
		// LL	(Rarg0), Rout
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		ll := mips.ALLV
		sc := mips.ASCV
		if v.Op == ssaop.OpMIPS64LoweredAtomicExchange32 {
			ll = mips.ALL
			sc = mips.ASC
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP
		p1 := s.Prog(ll)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(mips.ASYNC)
	case ssaop.OpMIPS64LoweredAtomicAdd32, ssaop.OpMIPS64LoweredAtomicAdd64:
		// SYNC
		// LL	(Rarg0), Rout
		// ADDV Rarg1, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDV Rarg1, Rout
		ll := mips.ALLV
		sc := mips.ASCV
		if v.Op == ssaop.OpMIPS64LoweredAtomicAdd32 {
			ll = mips.ALL
			sc = mips.ASC
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(ll)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(mips.AADDVU)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(mips.ASYNC)
		p4 := s.Prog(mips.AADDVU)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Args[1].Reg()
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()
	case ssaop.OpMIPS64LoweredAtomicAddconst32, ssaop.OpMIPS64LoweredAtomicAddconst64:
		// SYNC
		// LL	(Rarg0), Rout
		// ADDV $auxint, Rout, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		// ADDV $auxint, Rout
		ll := mips.ALLV
		sc := mips.ASCV
		if v.Op == ssaop.OpMIPS64LoweredAtomicAddconst32 {
			ll = mips.ALL
			sc = mips.ASC
		}
		s.Prog(mips.ASYNC)
		p := s.Prog(ll)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(mips.AADDVU)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = v.AuxInt
		p1.Reg = v.Reg0()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP
		p2 := s.Prog(sc)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()
		p3 := s.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
		s.Prog(mips.ASYNC)
		p4 := s.Prog(mips.AADDVU)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = v.AuxInt
		p4.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Reg0()
	case ssaop.OpMIPS64LoweredAtomicAnd32,
		ssaop.OpMIPS64LoweredAtomicOr32:
		// SYNC
		// LL	(Rarg0), Rtmp
		// AND/OR	Rarg1, Rtmp
		// SC	Rtmp, (Rarg0)
		// BEQ	Rtmp, -3(PC)
		// SYNC
		s.Prog(mips.ASYNC)

		p := s.Prog(mips.ALL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP

		p1 := s.Prog(v.Op.Asm())
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = v.Args[1].Reg()
		p1.Reg = mips.REGTMP
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP

		p2 := s.Prog(mips.ASC)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := s.Prog(mips.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = mips.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)

		s.Prog(mips.ASYNC)

	case ssaop.OpMIPS64LoweredAtomicCas32, ssaop.OpMIPS64LoweredAtomicCas64:
		// MOVV $0, Rout
		// SYNC
		// LL	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 4(PC)
		// MOVV Rarg2, Rout
		// SC	Rout, (Rarg0)
		// BEQ	Rout, -4(PC)
		// SYNC
		ll := mips.ALLV
		sc := mips.ASCV
		if v.Op == ssaop.OpMIPS64LoweredAtomicCas32 {
			ll = mips.ALL
			sc = mips.ASC
		}
		p := s.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		s.Prog(mips.ASYNC)
		p1 := s.Prog(ll)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = v.Args[0].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP
		p2 := s.Prog(mips.ABNE)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = mips.REGTMP
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(mips.AMOVV)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg0()
		p4 := s.Prog(sc)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = v.Reg0()
		p4.To.Type = obj.TYPE_MEM
		p4.To.Reg = v.Args[0].Reg()
		p5 := s.Prog(mips.ABEQ)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Reg0()
		p5.To.Type = obj.TYPE_BRANCH
		p5.To.SetTarget(p1)
		p6 := s.Prog(mips.ASYNC)
		p2.To.SetTarget(p6)
	case ssaop.OpMIPS64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(mips.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssaop.OpMIPS64FPFlagTrue,
		ssaop.OpMIPS64FPFlagFalse:
		// MOVV	$0, r
		// BFPF	2(PC)
		// MOVV	$1, r
		branch := mips.ABFPF
		if v.Op == ssaop.OpMIPS64FPFlagFalse {
			branch = mips.ABFPT
		}
		p := s.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = mips.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p2 := s.Prog(branch)
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(mips.AMOVV)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = 1
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Reg()
		p4 := s.Prog(obj.ANOP) // not a machine instruction, for branch to land
		p2.To.SetTarget(p4)
	case ssaop.OpMIPS64LoweredGetClosurePtr:
		// Closure pointer is R22 (mips.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpMIPS64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(mips.AMOVV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpMIPS64LoweredPubBarrier:
		// SYNC
		s.Prog(v.Op.Asm())
	case ssaop.OpClobber, ssaop.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var blockJump = map[block.BlockKind]struct {
	asm, invasm obj.As
}{
	block.BlockMIPS64EQ:  {mips.ABEQ, mips.ABNE},
	block.BlockMIPS64NE:  {mips.ABNE, mips.ABEQ},
	block.BlockMIPS64LTZ: {mips.ABLTZ, mips.ABGEZ},
	block.BlockMIPS64GEZ: {mips.ABGEZ, mips.ABLTZ},
	block.BlockMIPS64LEZ: {mips.ABLEZ, mips.ABGTZ},
	block.BlockMIPS64GTZ: {mips.ABGTZ, mips.ABLEZ},
	block.BlockMIPS64FPT: {mips.ABFPT, mips.ABFPF},
	block.BlockMIPS64FPF: {mips.ABFPF, mips.ABFPT},
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
	case block.BlockMIPS64EQ, block.BlockMIPS64NE,
		block.BlockMIPS64LTZ, block.BlockMIPS64GEZ,
		block.BlockMIPS64LEZ, block.BlockMIPS64GTZ,
		block.BlockMIPS64FPT, block.BlockMIPS64FPF:
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
	case block.BlockMIPS64JUMPTABLE:
		// SLLV	$3, Rarg0, Rtmp
		// ADDVU	Rarg1, Rtmp
		// MOVV	(Rtmp), Rtmp
		// JMP	(Rtmp)
		p := s.Prog(mips.ASLLV)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 3 // idx*8
		p.Reg = b.Controls[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = mips.REGTMP
		p1 := s.Prog(mips.AADDVU)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = b.Controls[1].Reg()
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = mips.REGTMP
		p2 := s.Prog(mips.AMOVV)
		p2.From.Type = obj.TYPE_MEM
		p2.From.Reg = mips.REGTMP
		p2.From.Offset = 0
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = mips.REGTMP
		p3 := s.Prog(obj.AJMP)
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = mips.REGTMP
		// Save jump tables for later resolution of the target blocks.
		s.JumpTables = append(s.JumpTables, b)
	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}
