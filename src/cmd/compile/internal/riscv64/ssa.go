// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64

import (
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
	"cmd/internal/obj/riscv"
	"internal/abi"
)

// ssaRegToReg maps ssa register numbers to obj register numbers.
var ssaRegToReg = []int16{
	riscv.REG_X0,
	// X1 (LR): unused
	riscv.REG_X2,
	riscv.REG_X3,
	riscv.REG_X4,
	riscv.REG_X5,
	riscv.REG_X6,
	riscv.REG_X7,
	riscv.REG_X8,
	riscv.REG_X9,
	riscv.REG_X10,
	riscv.REG_X11,
	riscv.REG_X12,
	riscv.REG_X13,
	riscv.REG_X14,
	riscv.REG_X15,
	riscv.REG_X16,
	riscv.REG_X17,
	riscv.REG_X18,
	riscv.REG_X19,
	riscv.REG_X20,
	riscv.REG_X21,
	riscv.REG_X22,
	riscv.REG_X23,
	riscv.REG_X24,
	riscv.REG_X25,
	riscv.REG_X26,
	riscv.REG_X27,
	riscv.REG_X28,
	riscv.REG_X29,
	riscv.REG_X30,
	riscv.REG_X31,
	riscv.REG_F0,
	riscv.REG_F1,
	riscv.REG_F2,
	riscv.REG_F3,
	riscv.REG_F4,
	riscv.REG_F5,
	riscv.REG_F6,
	riscv.REG_F7,
	riscv.REG_F8,
	riscv.REG_F9,
	riscv.REG_F10,
	riscv.REG_F11,
	riscv.REG_F12,
	riscv.REG_F13,
	riscv.REG_F14,
	riscv.REG_F15,
	riscv.REG_F16,
	riscv.REG_F17,
	riscv.REG_F18,
	riscv.REG_F19,
	riscv.REG_F20,
	riscv.REG_F21,
	riscv.REG_F22,
	riscv.REG_F23,
	riscv.REG_F24,
	riscv.REG_F25,
	riscv.REG_F26,
	riscv.REG_F27,
	riscv.REG_F28,
	riscv.REG_F29,
	riscv.REG_F30,
	riscv.REG_F31,
	0, // SB isn't a real register.  We fill an Addr.Reg field with 0 in this case.
}

func loadByType(t *types.Type) obj.As {
	width := t.Size()

	if t.IsFloat() {
		switch width {
		case 4:
			return riscv.AMOVF
		case 8:
			return riscv.AMOVD
		default:
			base.Fatalf("unknown float width for load %d in type %v", width, t)
			return 0
		}
	}

	switch width {
	case 1:
		if t.IsSigned() {
			return riscv.AMOVB
		} else {
			return riscv.AMOVBU
		}
	case 2:
		if t.IsSigned() {
			return riscv.AMOVH
		} else {
			return riscv.AMOVHU
		}
	case 4:
		if t.IsSigned() {
			return riscv.AMOVW
		} else {
			return riscv.AMOVWU
		}
	case 8:
		return riscv.AMOV
	default:
		base.Fatalf("unknown width for load %d in type %v", width, t)
		return 0
	}
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	width := t.Size()

	if t.IsFloat() {
		switch width {
		case 4:
			return riscv.AMOVF
		case 8:
			return riscv.AMOVD
		default:
			base.Fatalf("unknown float width for store %d in type %v", width, t)
			return 0
		}
	}

	switch width {
	case 1:
		return riscv.AMOVB
	case 2:
		return riscv.AMOVH
	case 4:
		return riscv.AMOVW
	case 8:
		return riscv.AMOV
	default:
		base.Fatalf("unknown width for store %d in type %v", width, t)
		return 0
	}
}

// largestMove returns the largest move instruction possible and its size,
// given the alignment of the total size of the move.
//
// e.g., a 16-byte move may use MOV, but an 11-byte move must use MOVB.
//
// Note that the moves may not be on naturally aligned addresses depending on
// the source and destination.
//
// This matches the calculation in ssa.moveSize.
func largestMove(alignment int64) (obj.As, int64) {
	switch {
	case alignment%8 == 0:
		return riscv.AMOV, 8
	case alignment%4 == 0:
		return riscv.AMOVW, 4
	case alignment%2 == 0:
		return riscv.AMOVH, 2
	default:
		return riscv.AMOVB, 1
	}
}

var fracMovOps = []obj.As{riscv.AMOVB, riscv.AMOVH, riscv.AMOVW, riscv.AMOV}

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
// RISC-V has no flags, so this is a no-op.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	s.SetPos(v.Pos)

	switch v.Op {
	case ssaop.OpInitMem:
		// memory arg needs no code
	case ssaop.OpArg:
		// input args need no code
	case ssaop.OpPhi:
		ssagen.CheckLoweredPhi(v)
	case ssaop.OpCopy, ssaop.OpRISCV64MOVDreg:
		if v.Type.IsMemory() {
			return
		}
		rs := v.Args[0].Reg()
		rd := v.Reg()
		if rs == rd {
			return
		}
		as := riscv.AMOV
		if v.Type.IsFloat() {
			as = riscv.AMOVD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = rs
		p.To.Type = obj.TYPE_REG
		p.To.Reg = rd
	case ssaop.OpRISCV64MOVDnop:
		// nothing to do
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(loadByType(v.Type))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddrAuto(&p.To, v)
	case ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		for _, a := range v.Block.Func.RegArgs {
			// Pass the spill/unspill information along to the assembler, offset by size of
			// the saved LR slot.
			addr := ssagen.SpillSlotAddr(a, riscv.REG_SP, base.Ctxt.Arch.FixedFrameSize)
			s.FuncInfo().AddSpill(
				obj.RegSpill{Reg: a.Reg, Addr: addr, Unspill: loadByType(a.Type), Spill: storeByType(a.Type)})
		}
		v.Block.Func.RegArgs = nil

		ssagen.CheckArgReg(v)
	case ssaop.OpSP, ssaop.OpSB, ssaop.OpGetG:
		// nothing to do
	case ssaop.OpRISCV64MOVBreg, ssaop.OpRISCV64MOVHreg, ssaop.OpRISCV64MOVWreg,
		ssaop.OpRISCV64MOVBUreg, ssaop.OpRISCV64MOVHUreg, ssaop.OpRISCV64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssaop.OpCopy || a.Op == ssaop.OpRISCV64MOVDreg {
			a = a.Args[0]
		}
		as := v.Op.Asm()
		rs := v.Args[0].Reg()
		rd := v.Reg()
		if a.Op == ssaop.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssaop.OpRISCV64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssaop.OpRISCV64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssaop.OpRISCV64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssaop.OpRISCV64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssaop.OpRISCV64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssaop.OpRISCV64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load and already sign/zero-extended
				if rs == rd {
					return
				}
				as = riscv.AMOV
			default:
			}
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = rs
		p.To.Type = obj.TYPE_REG
		p.To.Reg = rd
	case ssaop.OpRISCV64ADD, ssaop.OpRISCV64SUB, ssaop.OpRISCV64SUBW, ssaop.OpRISCV64XNOR, ssaop.OpRISCV64XOR,
		ssaop.OpRISCV64OR, ssaop.OpRISCV64ORN, ssaop.OpRISCV64AND, ssaop.OpRISCV64ANDN,
		ssaop.OpRISCV64SLL, ssaop.OpRISCV64SLLW, ssaop.OpRISCV64SRA, ssaop.OpRISCV64SRAW, ssaop.OpRISCV64SRL, ssaop.OpRISCV64SRLW,
		ssaop.OpRISCV64SLT, ssaop.OpRISCV64SLTU, ssaop.OpRISCV64MUL, ssaop.OpRISCV64MULW, ssaop.OpRISCV64MULH,
		ssaop.OpRISCV64MULHU, ssaop.OpRISCV64DIV, ssaop.OpRISCV64DIVU, ssaop.OpRISCV64DIVW,
		ssaop.OpRISCV64DIVUW, ssaop.OpRISCV64REM, ssaop.OpRISCV64REMU, ssaop.OpRISCV64REMW,
		ssaop.OpRISCV64REMUW,
		ssaop.OpRISCV64ROL, ssaop.OpRISCV64ROLW, ssaop.OpRISCV64ROR, ssaop.OpRISCV64RORW,
		ssaop.OpRISCV64FADDS, ssaop.OpRISCV64FSUBS, ssaop.OpRISCV64FMULS, ssaop.OpRISCV64FDIVS,
		ssaop.OpRISCV64FEQS, ssaop.OpRISCV64FNES, ssaop.OpRISCV64FLTS, ssaop.OpRISCV64FLES,
		ssaop.OpRISCV64FADDD, ssaop.OpRISCV64FSUBD, ssaop.OpRISCV64FMULD, ssaop.OpRISCV64FDIVD,
		ssaop.OpRISCV64FEQD, ssaop.OpRISCV64FNED, ssaop.OpRISCV64FLTD, ssaop.OpRISCV64FLED, ssaop.OpRISCV64FSGNJD,
		ssaop.OpRISCV64MIN, ssaop.OpRISCV64MAX, ssaop.OpRISCV64MINU, ssaop.OpRISCV64MAXU,
		ssaop.OpRISCV64SH1ADD, ssaop.OpRISCV64SH2ADD, ssaop.OpRISCV64SH3ADD,
		ssaop.OpRISCV64CZEROEQZ, ssaop.OpRISCV64CZERONEZ:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r

	case ssaop.OpRISCV64LoweredFMAXD, ssaop.OpRISCV64LoweredFMIND, ssaop.OpRISCV64LoweredFMAXS, ssaop.OpRISCV64LoweredFMINS:
		// Most of FMIN/FMAX result match Go's required behaviour, unless one of the
		// inputs is a NaN. As such, we need to explicitly test for NaN
		// before using FMIN/FMAX.

		// FADD Rarg0, Rarg1, Rout // FADD is used to propagate a NaN to the result in these cases.
		// FEQ  Rarg0, Rarg0, Rtmp
		// BEQZ Rtmp, end
		// FEQ  Rarg1, Rarg1, Rtmp
		// BEQZ Rtmp, end
		// F(MIN | MAX)

		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg()
		add, feq := riscv.AFADDD, riscv.AFEQD
		if v.Op == ssaop.OpRISCV64LoweredFMAXS || v.Op == ssaop.OpRISCV64LoweredFMINS {
			add = riscv.AFADDS
			feq = riscv.AFEQS
		}

		p1 := s.Prog(add)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r0
		p1.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out

		p2 := s.Prog(feq)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = r0
		p2.Reg = r0
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = riscv.REG_TMP

		p3 := s.Prog(riscv.ABEQ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = riscv.REG_ZERO
		p3.Reg = riscv.REG_TMP
		p3.To.Type = obj.TYPE_BRANCH

		p4 := s.Prog(feq)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = r1
		p4.Reg = r1
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = riscv.REG_TMP

		p5 := s.Prog(riscv.ABEQ)
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = riscv.REG_ZERO
		p5.Reg = riscv.REG_TMP
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

	case ssaop.OpRISCV64LoweredMuluhilo:
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		p := s.Prog(riscv.AMULHU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		p1 := s.Prog(riscv.AMUL)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = r0
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg1()
	case ssaop.OpRISCV64LoweredMuluover:
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		p := s.Prog(riscv.AMULHU)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg1()
		p1 := s.Prog(riscv.AMUL)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = r0
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = v.Reg0()
		p2 := s.Prog(riscv.ASNEZ)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Reg1()
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg1()
	case ssaop.OpRISCV64FMADDD, ssaop.OpRISCV64FMSUBD, ssaop.OpRISCV64FNMADDD, ssaop.OpRISCV64FNMSUBD,
		ssaop.OpRISCV64FMADDS, ssaop.OpRISCV64FMSUBS, ssaop.OpRISCV64FNMADDS, ssaop.OpRISCV64FNMSUBS:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		r3 := v.Args[2].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.AddRestSource(obj.Addr{Type: obj.TYPE_REG, Reg: r3})
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpRISCV64FSQRTS, ssaop.OpRISCV64FSQRTD,
		ssaop.OpRISCV64FNEGS, ssaop.OpRISCV64FNEGD,
		ssaop.OpRISCV64FABSS, ssaop.OpRISCV64FABSD,
		ssaop.OpRISCV64FMVSX, ssaop.OpRISCV64FMVXS, ssaop.OpRISCV64FMVDX, ssaop.OpRISCV64FMVXD,
		ssaop.OpRISCV64FCVTSW, ssaop.OpRISCV64FCVTSL, ssaop.OpRISCV64FCVTWS, ssaop.OpRISCV64FCVTLS,
		ssaop.OpRISCV64FCVTDW, ssaop.OpRISCV64FCVTDL, ssaop.OpRISCV64FCVTWD, ssaop.OpRISCV64FCVTLD, ssaop.OpRISCV64FCVTDS, ssaop.OpRISCV64FCVTSD,
		ssaop.OpRISCV64FCLASSS, ssaop.OpRISCV64FCLASSD,
		ssaop.OpRISCV64NOT, ssaop.OpRISCV64NEG, ssaop.OpRISCV64NEGW, ssaop.OpRISCV64CLZ, ssaop.OpRISCV64CLZW, ssaop.OpRISCV64CTZ, ssaop.OpRISCV64CTZW,
		ssaop.OpRISCV64REV8, ssaop.OpRISCV64CPOP, ssaop.OpRISCV64CPOPW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64ADDI, ssaop.OpRISCV64ADDIW, ssaop.OpRISCV64XORI, ssaop.OpRISCV64ORI, ssaop.OpRISCV64ANDI,
		ssaop.OpRISCV64SLLI, ssaop.OpRISCV64SLLIW, ssaop.OpRISCV64SRAI, ssaop.OpRISCV64SRAIW,
		ssaop.OpRISCV64SRLI, ssaop.OpRISCV64SRLIW, ssaop.OpRISCV64SLTI, ssaop.OpRISCV64SLTIU,
		ssaop.OpRISCV64RORI, ssaop.OpRISCV64RORIW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64FMOVDconst, ssaop.OpRISCV64FMOVFconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = v.AuxFloat()
		p.From.Name = obj.NAME_NONE
		p.From.Reg = obj.REG_NONE
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64MOVaddr:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_ADDR
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		var wantreg string
		// MOVW $sym+off(base), R
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
			// No sym, just MOVW $off(SP), R
			wantreg = "SP"
			p.From.Reg = riscv.REG_SP
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
	case ssaop.OpRISCV64MOVBload, ssaop.OpRISCV64MOVHload, ssaop.OpRISCV64MOVWload, ssaop.OpRISCV64MOVDload,
		ssaop.OpRISCV64MOVBUload, ssaop.OpRISCV64MOVHUload, ssaop.OpRISCV64MOVWUload,
		ssaop.OpRISCV64FMOVWload, ssaop.OpRISCV64FMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64MOVBstore, ssaop.OpRISCV64MOVHstore, ssaop.OpRISCV64MOVWstore, ssaop.OpRISCV64MOVDstore,
		ssaop.OpRISCV64FMOVWstore, ssaop.OpRISCV64FMOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpRISCV64MOVBstorezero, ssaop.OpRISCV64MOVHstorezero, ssaop.OpRISCV64MOVWstorezero, ssaop.OpRISCV64MOVDstorezero:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = riscv.REG_ZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpRISCV64SEQZ, ssaop.OpRISCV64SNEZ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpRISCV64CALLstatic, ssaop.OpRISCV64CALLclosure, ssaop.OpRISCV64CALLinter:
		s.Call(v)
	case ssaop.OpRISCV64CALLtail, ssaop.OpRISCV64CALLtailinter:
		s.TailCall(v)
	case ssaop.OpRISCV64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpRISCV64LoweredPanicBoundsRR, ssaop.OpRISCV64LoweredPanicBoundsRC, ssaop.OpRISCV64LoweredPanicBoundsCR, ssaop.OpRISCV64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpRISCV64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - riscv.REG_X5)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - riscv.REG_X5)
		case ssaop.OpRISCV64LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - riscv.REG_X5)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(riscv.AMOV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = riscv.REG_X5 + int16(yVal)
			}
		case ssaop.OpRISCV64LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - riscv.REG_X5)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(riscv.AMOV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = riscv.REG_X5 + int16(xVal)
			}
		case ssaop.OpRISCV64LoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(riscv.AMOV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = riscv.REG_X5 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(riscv.AMOV)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = riscv.REG_X5 + int16(yVal)
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

	case ssaop.OpRISCV64LoweredAtomicLoad8:
		p1 := s.Prog(riscv.AFENCE)
		p1.From.Type = obj.TYPE_SPECIAL
		p1.From.Offset = int64(riscv.SPOP_FENCE_RW)
		p1.To.Type = obj.TYPE_SPECIAL
		p1.To.Offset = int64(riscv.SPOP_FENCE_RW)

		p := s.Prog(riscv.AMOVBU)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

		p2 := s.Prog(riscv.AFENCE)
		p2.From.Type = obj.TYPE_SPECIAL
		p2.From.Offset = int64(riscv.SPOP_FENCE_R)
		p2.To.Type = obj.TYPE_SPECIAL
		p2.To.Offset = int64(riscv.SPOP_FENCE_RW)

	case ssaop.OpRISCV64LoweredAtomicLoad32, ssaop.OpRISCV64LoweredAtomicLoad64:
		as := riscv.ALRW
		if v.Op == ssaop.OpRISCV64LoweredAtomicLoad64 {
			as = riscv.ALRD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssaop.OpRISCV64LoweredAtomicStore8:
		p1 := s.Prog(riscv.AFENCE)
		p1.From.Type = obj.TYPE_SPECIAL
		p1.From.Offset = int64(riscv.SPOP_FENCE_RW)
		p1.To.Type = obj.TYPE_SPECIAL
		p1.To.Offset = int64(riscv.SPOP_FENCE_W)

		p := s.Prog(riscv.AMOVB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()

		p2 := s.Prog(riscv.AFENCE)
		p2.From.Type = obj.TYPE_SPECIAL
		p2.From.Offset = int64(riscv.SPOP_FENCE_RW)
		p2.To.Type = obj.TYPE_SPECIAL
		p2.To.Offset = int64(riscv.SPOP_FENCE_RW)

	case ssaop.OpRISCV64LoweredAtomicStore32, ssaop.OpRISCV64LoweredAtomicStore64:
		as := riscv.AAMOSWAPW
		if v.Op == ssaop.OpRISCV64LoweredAtomicStore64 {
			as = riscv.AAMOSWAPD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = riscv.REG_ZERO

	case ssaop.OpRISCV64LoweredAtomicAdd32, ssaop.OpRISCV64LoweredAtomicAdd64:
		as := riscv.AAMOADDW
		if v.Op == ssaop.OpRISCV64LoweredAtomicAdd64 {
			as = riscv.AAMOADDD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = riscv.REG_TMP

		p2 := s.Prog(riscv.AADD)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = riscv.REG_TMP
		p2.Reg = v.Args[1].Reg()
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Reg0()

	case ssaop.OpRISCV64LoweredAtomicExchange32, ssaop.OpRISCV64LoweredAtomicExchange64:
		as := riscv.AAMOSWAPW
		if v.Op == ssaop.OpRISCV64LoweredAtomicExchange64 {
			as = riscv.AAMOSWAPD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssaop.OpRISCV64LoweredAtomicCas32, ssaop.OpRISCV64LoweredAtomicCas64:
		// MOV  ZERO, Rout
		// LR	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 3(PC)
		// SC	Rarg2, (Rarg0), Rtmp
		// BNE	Rtmp, ZERO, -3(PC)
		// MOV	$1, Rout

		lr := riscv.ALRW
		sc := riscv.ASCW
		if v.Op == ssaop.OpRISCV64LoweredAtomicCas64 {
			lr = riscv.ALRD
			sc = riscv.ASCD
		}

		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()

		p := s.Prog(riscv.AMOV)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = riscv.REG_ZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out

		p1 := s.Prog(lr)
		p1.From.Type = obj.TYPE_MEM
		p1.From.Reg = r0
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = riscv.REG_TMP

		p2 := s.Prog(riscv.ABNE)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = r1
		p2.Reg = riscv.REG_TMP
		p2.To.Type = obj.TYPE_BRANCH

		p3 := s.Prog(sc)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = r2
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = r0
		p3.RegTo2 = riscv.REG_TMP

		p4 := s.Prog(riscv.ABNE)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = riscv.REG_TMP
		p4.Reg = riscv.REG_ZERO
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p1)

		p5 := s.Prog(riscv.AMOV)
		p5.From.Type = obj.TYPE_CONST
		p5.From.Offset = 1
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = out

		p6 := s.Prog(obj.ANOP)
		p2.To.SetTarget(p6)

	case ssaop.OpRISCV64LoweredAtomicAnd32, ssaop.OpRISCV64LoweredAtomicOr32:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = riscv.REG_ZERO

	case ssaop.OpRISCV64LoweredAtomicAnd32value, ssaop.OpRISCV64LoweredAtomicAnd64value,
		ssaop.OpRISCV64LoweredAtomicOr32value, ssaop.OpRISCV64LoweredAtomicOr64value:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssaop.OpRISCV64LoweredZero:
		ptr := v.Args[0].Reg()
		sc := v.AuxValAndOff()
		n := sc.Val64()

		mov, sz := largestMove(sc.Off64())

		// mov	ZERO, (offset)(Rarg0)
		var off int64
		for n >= sz {
			zeroOp(s, mov, ptr, off)
			off += sz
			n -= sz
		}

		for i := len(fracMovOps) - 1; i >= 0; i-- {
			tsz := int64(1 << i)
			if n < tsz {
				continue
			}
			zeroOp(s, fracMovOps[i], ptr, off)
			off += tsz
			n -= tsz
		}

	case ssaop.OpRISCV64LoweredZeroLoop:
		ptr := v.Args[0].Reg()
		sc := v.AuxValAndOff()
		n := sc.Val64()
		mov, sz := largestMove(sc.Off64())
		chunk := 8 * sz

		if n <= 3*chunk {
			v.Fatalf("ZeroLoop too small:%d, expect:%d", n, 3*chunk)
		}

		tmp := v.RegTmp()

		p := s.Prog(riscv.AADD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n - n%chunk
		p.Reg = ptr
		p.To.Type = obj.TYPE_REG
		p.To.Reg = tmp

		for i := int64(0); i < 8; i++ {
			zeroOp(s, mov, ptr, sz*i)
		}

		p2 := s.Prog(riscv.AADD)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = chunk
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = ptr

		p3 := s.Prog(riscv.ABNE)
		p3.From.Reg = tmp
		p3.From.Type = obj.TYPE_REG
		p3.Reg = ptr
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p.Link)

		n %= chunk

		// mov	ZERO, (offset)(Rarg0)
		var off int64
		for n >= sz {
			zeroOp(s, mov, ptr, off)
			off += sz
			n -= sz
		}

		for i := len(fracMovOps) - 1; i >= 0; i-- {
			tsz := int64(1 << i)
			if n < tsz {
				continue
			}
			zeroOp(s, fracMovOps[i], ptr, off)
			off += tsz
			n -= tsz
		}

	case ssaop.OpRISCV64LoweredMove:
		dst := v.Args[0].Reg()
		src := v.Args[1].Reg()
		if dst == src {
			break
		}

		sa := v.AuxValAndOff()
		n := sa.Val64()
		mov, sz := largestMove(sa.Off64())

		var off int64
		tmp := int16(riscv.REG_X5)
		for n >= sz {
			moveOp(s, mov, dst, src, tmp, off)
			off += sz
			n -= sz
		}

		for i := len(fracMovOps) - 1; i >= 0; i-- {
			tsz := int64(1 << i)
			if n < tsz {
				continue
			}
			moveOp(s, fracMovOps[i], dst, src, tmp, off)
			off += tsz
			n -= tsz
		}

	case ssaop.OpRISCV64LoweredMoveLoop:
		dst := v.Args[0].Reg()
		src := v.Args[1].Reg()
		if dst == src {
			break
		}

		sc := v.AuxValAndOff()
		n := sc.Val64()
		mov, sz := largestMove(sc.Off64())
		chunk := 8 * sz

		if n <= 3*chunk {
			v.Fatalf("MoveLoop too small:%d, expect:%d", n, 3*chunk)
		}
		tmp := int16(riscv.REG_X5)

		p := s.Prog(riscv.AADD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n - n%chunk
		p.Reg = src
		p.To.Type = obj.TYPE_REG
		p.To.Reg = riscv.REG_X6

		for i := int64(0); i < 8; i++ {
			moveOp(s, mov, dst, src, tmp, sz*i)
		}

		p1 := s.Prog(riscv.AADD)
		p1.From.Type = obj.TYPE_CONST
		p1.From.Offset = chunk
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = src

		p2 := s.Prog(riscv.AADD)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = chunk
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = dst

		p3 := s.Prog(riscv.ABNE)
		p3.From.Reg = riscv.REG_X6
		p3.From.Type = obj.TYPE_REG
		p3.Reg = src
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p.Link)

		n %= chunk

		var off int64
		for n >= sz {
			moveOp(s, mov, dst, src, tmp, off)
			off += sz
			n -= sz
		}

		for i := len(fracMovOps) - 1; i >= 0; i-- {
			tsz := int64(1 << i)
			if n < tsz {
				continue
			}
			moveOp(s, fracMovOps[i], dst, src, tmp, off)
			off += tsz
			n -= tsz
		}

	case ssaop.OpRISCV64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(riscv.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = riscv.REG_ZERO
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos == 1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}

	case ssaop.OpRISCV64LoweredGetClosurePtr:
		// Closure pointer is S10 (riscv.REG_CTXT).
		ssagen.CheckLoweredGetClosurePtr(v)

	case ssaop.OpRISCV64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(riscv.AMOV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpRISCV64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssaop.OpRISCV64LoweredPubBarrier:
		// FENCE W, W
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_SPECIAL
		p.From.Offset = int64(riscv.SPOP_FENCE_W)
		p.To.Type = obj.TYPE_SPECIAL
		p.To.Offset = int64(riscv.SPOP_FENCE_W)

	case ssaop.OpRISCV64LoweredRound32F, ssaop.OpRISCV64LoweredRound64F:
		// input is already rounded

	case ssaop.OpClobber, ssaop.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.

	default:
		v.Fatalf("Unhandled op %v", v.Op)
	}
}

var blockBranch = [...]obj.As{
	block.BlockRISCV64BEQ:  riscv.ABEQ,
	block.BlockRISCV64BEQZ: riscv.ABEQZ,
	block.BlockRISCV64BGE:  riscv.ABGE,
	block.BlockRISCV64BGEU: riscv.ABGEU,
	block.BlockRISCV64BGEZ: riscv.ABGEZ,
	block.BlockRISCV64BGTZ: riscv.ABGTZ,
	block.BlockRISCV64BLEZ: riscv.ABLEZ,
	block.BlockRISCV64BLT:  riscv.ABLT,
	block.BlockRISCV64BLTU: riscv.ABLTU,
	block.BlockRISCV64BLTZ: riscv.ABLTZ,
	block.BlockRISCV64BNE:  riscv.ABNE,
	block.BlockRISCV64BNEZ: riscv.ABNEZ,
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	s.SetPos(b.Pos)

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
	case block.BlockRISCV64BEQ, block.BlockRISCV64BEQZ, block.BlockRISCV64BNE, block.BlockRISCV64BNEZ,
		block.BlockRISCV64BLT, block.BlockRISCV64BLEZ, block.BlockRISCV64BGE, block.BlockRISCV64BGEZ,
		block.BlockRISCV64BLTZ, block.BlockRISCV64BGTZ, block.BlockRISCV64BLTU, block.BlockRISCV64BGEU:

		as := blockBranch[b.Kind]
		invAs := riscv.InvertBranch(as)

		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = s.Br(invAs, b.Succs[1].Block())
		case b.Succs[1].Block():
			p = s.Br(as, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				p = s.Br(as, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				p = s.Br(invAs, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}

		p.From.Type = obj.TYPE_REG
		switch b.Kind {
		case block.BlockRISCV64BEQ, block.BlockRISCV64BNE, block.BlockRISCV64BLT, block.BlockRISCV64BGE, block.BlockRISCV64BLTU, block.BlockRISCV64BGEU:
			if b.NumControls() != 2 {
				b.Fatalf("Unexpected number of controls (%d != 2): %s", b.NumControls(), b.LongString())
			}
			p.From.Reg = b.Controls[0].Reg()
			p.Reg = b.Controls[1].Reg()

		case block.BlockRISCV64BEQZ, block.BlockRISCV64BNEZ, block.BlockRISCV64BGEZ, block.BlockRISCV64BLEZ, block.BlockRISCV64BLTZ, block.BlockRISCV64BGTZ:
			if b.NumControls() != 1 {
				b.Fatalf("Unexpected number of controls (%d != 1): %s", b.NumControls(), b.LongString())
			}
			p.From.Reg = b.Controls[0].Reg()
		}

	default:
		b.Fatalf("Unhandled block: %s", b.LongString())
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

func zeroOp(s *ssagen.State, mov obj.As, reg int16, off int64) {
	p := s.Prog(mov)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = riscv.REG_ZERO
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
	return
}

func moveOp(s *ssagen.State, mov obj.As, dst int16, src int16, tmp int16, off int64) {
	p := s.Prog(mov)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = src
	p.From.Offset = off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp

	p1 := s.Prog(mov)
	p1.From.Type = obj.TYPE_REG
	p1.From.Reg = tmp
	p1.To.Type = obj.TYPE_MEM
	p1.To.Reg = dst
	p1.To.Offset = off

	return
}
