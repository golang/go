// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/riscv"
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

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
// RISC-V has no flags, so this is a no-op.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	s.SetPos(v.Pos)

	switch v.Op {
	case ssa.OpInitMem:
		// memory arg needs no code
	case ssa.OpArg:
		// input args need no code
	case ssa.OpPhi:
		ssagen.CheckLoweredPhi(v)
	case ssa.OpCopy, ssa.OpRISCV64MOVDreg:
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
	case ssa.OpRISCV64MOVDnop:
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
	case ssa.OpSP, ssa.OpSB, ssa.OpGetG:
		// nothing to do
	case ssa.OpRISCV64MOVBreg, ssa.OpRISCV64MOVHreg, ssa.OpRISCV64MOVWreg,
		ssa.OpRISCV64MOVBUreg, ssa.OpRISCV64MOVHUreg, ssa.OpRISCV64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssa.OpCopy || a.Op == ssa.OpRISCV64MOVDreg {
			a = a.Args[0]
		}
		as := v.Op.Asm()
		rs := v.Args[0].Reg()
		rd := v.Reg()
		if a.Op == ssa.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssa.OpRISCV64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssa.OpRISCV64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssa.OpRISCV64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssa.OpRISCV64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssa.OpRISCV64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssa.OpRISCV64MOVWUreg && t.Size() == 4 && !t.IsSigned():
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
	case ssa.OpRISCV64ADD, ssa.OpRISCV64SUB, ssa.OpRISCV64SUBW, ssa.OpRISCV64XOR, ssa.OpRISCV64OR, ssa.OpRISCV64AND,
		ssa.OpRISCV64SLL, ssa.OpRISCV64SRA, ssa.OpRISCV64SRAW, ssa.OpRISCV64SRL, ssa.OpRISCV64SRLW,
		ssa.OpRISCV64SLT, ssa.OpRISCV64SLTU, ssa.OpRISCV64MUL, ssa.OpRISCV64MULW, ssa.OpRISCV64MULH,
		ssa.OpRISCV64MULHU, ssa.OpRISCV64DIV, ssa.OpRISCV64DIVU, ssa.OpRISCV64DIVW,
		ssa.OpRISCV64DIVUW, ssa.OpRISCV64REM, ssa.OpRISCV64REMU, ssa.OpRISCV64REMW,
		ssa.OpRISCV64REMUW,
		ssa.OpRISCV64FADDS, ssa.OpRISCV64FSUBS, ssa.OpRISCV64FMULS, ssa.OpRISCV64FDIVS,
		ssa.OpRISCV64FEQS, ssa.OpRISCV64FNES, ssa.OpRISCV64FLTS, ssa.OpRISCV64FLES,
		ssa.OpRISCV64FADDD, ssa.OpRISCV64FSUBD, ssa.OpRISCV64FMULD, ssa.OpRISCV64FDIVD,
		ssa.OpRISCV64FEQD, ssa.OpRISCV64FNED, ssa.OpRISCV64FLTD, ssa.OpRISCV64FLED,
		ssa.OpRISCV64FSGNJD:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpRISCV64LoweredMuluhilo:
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
	case ssa.OpRISCV64LoweredMuluover:
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
	case ssa.OpRISCV64FMADDD, ssa.OpRISCV64FMSUBD, ssa.OpRISCV64FNMADDD, ssa.OpRISCV64FNMSUBD,
		ssa.OpRISCV64FMADDS, ssa.OpRISCV64FMSUBS, ssa.OpRISCV64FNMADDS, ssa.OpRISCV64FNMSUBS:
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
	case ssa.OpRISCV64FSQRTS, ssa.OpRISCV64FNEGS, ssa.OpRISCV64FABSD, ssa.OpRISCV64FSQRTD, ssa.OpRISCV64FNEGD,
		ssa.OpRISCV64FMVSX, ssa.OpRISCV64FMVDX,
		ssa.OpRISCV64FCVTSW, ssa.OpRISCV64FCVTSL, ssa.OpRISCV64FCVTWS, ssa.OpRISCV64FCVTLS,
		ssa.OpRISCV64FCVTDW, ssa.OpRISCV64FCVTDL, ssa.OpRISCV64FCVTWD, ssa.OpRISCV64FCVTLD, ssa.OpRISCV64FCVTDS, ssa.OpRISCV64FCVTSD,
		ssa.OpRISCV64NOT, ssa.OpRISCV64NEG, ssa.OpRISCV64NEGW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpRISCV64ADDI, ssa.OpRISCV64ADDIW, ssa.OpRISCV64XORI, ssa.OpRISCV64ORI, ssa.OpRISCV64ANDI,
		ssa.OpRISCV64SLLI, ssa.OpRISCV64SRAI, ssa.OpRISCV64SRAIW, ssa.OpRISCV64SRLI, ssa.OpRISCV64SRLIW, ssa.OpRISCV64SLTI,
		ssa.OpRISCV64SLTIU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpRISCV64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpRISCV64MOVaddr:
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
	case ssa.OpRISCV64MOVBload, ssa.OpRISCV64MOVHload, ssa.OpRISCV64MOVWload, ssa.OpRISCV64MOVDload,
		ssa.OpRISCV64MOVBUload, ssa.OpRISCV64MOVHUload, ssa.OpRISCV64MOVWUload,
		ssa.OpRISCV64FMOVWload, ssa.OpRISCV64FMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpRISCV64MOVBstore, ssa.OpRISCV64MOVHstore, ssa.OpRISCV64MOVWstore, ssa.OpRISCV64MOVDstore,
		ssa.OpRISCV64FMOVWstore, ssa.OpRISCV64FMOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpRISCV64MOVBstorezero, ssa.OpRISCV64MOVHstorezero, ssa.OpRISCV64MOVWstorezero, ssa.OpRISCV64MOVDstorezero:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = riscv.REG_ZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpRISCV64SEQZ, ssa.OpRISCV64SNEZ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpRISCV64CALLstatic, ssa.OpRISCV64CALLclosure, ssa.OpRISCV64CALLinter:
		s.Call(v)
	case ssa.OpRISCV64CALLtail:
		s.TailCall(v)
	case ssa.OpRISCV64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]
	case ssa.OpRISCV64LoweredPanicBoundsA, ssa.OpRISCV64LoweredPanicBoundsB, ssa.OpRISCV64LoweredPanicBoundsC:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ssagen.BoundsCheckFunc[v.AuxInt]
		s.UseArgs(16) // space used in callee args area by assembly stubs

	case ssa.OpRISCV64LoweredAtomicLoad8:
		s.Prog(riscv.AFENCE)
		p := s.Prog(riscv.AMOVBU)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
		s.Prog(riscv.AFENCE)

	case ssa.OpRISCV64LoweredAtomicLoad32, ssa.OpRISCV64LoweredAtomicLoad64:
		as := riscv.ALRW
		if v.Op == ssa.OpRISCV64LoweredAtomicLoad64 {
			as = riscv.ALRD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()

	case ssa.OpRISCV64LoweredAtomicStore8:
		s.Prog(riscv.AFENCE)
		p := s.Prog(riscv.AMOVB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		s.Prog(riscv.AFENCE)

	case ssa.OpRISCV64LoweredAtomicStore32, ssa.OpRISCV64LoweredAtomicStore64:
		as := riscv.AAMOSWAPW
		if v.Op == ssa.OpRISCV64LoweredAtomicStore64 {
			as = riscv.AAMOSWAPD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = riscv.REG_ZERO

	case ssa.OpRISCV64LoweredAtomicAdd32, ssa.OpRISCV64LoweredAtomicAdd64:
		as := riscv.AAMOADDW
		if v.Op == ssa.OpRISCV64LoweredAtomicAdd64 {
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

	case ssa.OpRISCV64LoweredAtomicExchange32, ssa.OpRISCV64LoweredAtomicExchange64:
		as := riscv.AAMOSWAPW
		if v.Op == ssa.OpRISCV64LoweredAtomicExchange64 {
			as = riscv.AAMOSWAPD
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = v.Reg0()

	case ssa.OpRISCV64LoweredAtomicCas32, ssa.OpRISCV64LoweredAtomicCas64:
		// MOV  ZERO, Rout
		// LR	(Rarg0), Rtmp
		// BNE	Rtmp, Rarg1, 3(PC)
		// SC	Rarg2, (Rarg0), Rtmp
		// BNE	Rtmp, ZERO, -3(PC)
		// MOV	$1, Rout

		lr := riscv.ALRW
		sc := riscv.ASCW
		if v.Op == ssa.OpRISCV64LoweredAtomicCas64 {
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

	case ssa.OpRISCV64LoweredAtomicAnd32, ssa.OpRISCV64LoweredAtomicOr32:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.RegTo2 = riscv.REG_ZERO

	case ssa.OpRISCV64LoweredZero:
		mov, sz := largestMove(v.AuxInt)

		//	mov	ZERO, (Rarg0)
		//	ADD	$sz, Rarg0
		//	BGEU	Rarg1, Rarg0, -2(PC)

		p := s.Prog(mov)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = riscv.REG_ZERO
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()

		p2 := s.Prog(riscv.AADD)
		p2.From.Type = obj.TYPE_CONST
		p2.From.Offset = sz
		p2.To.Type = obj.TYPE_REG
		p2.To.Reg = v.Args[0].Reg()

		p3 := s.Prog(riscv.ABGEU)
		p3.To.Type = obj.TYPE_BRANCH
		p3.Reg = v.Args[0].Reg()
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[1].Reg()
		p3.To.SetTarget(p)

	case ssa.OpRISCV64LoweredMove:
		mov, sz := largestMove(v.AuxInt)

		//	mov	(Rarg1), T2
		//	mov	T2, (Rarg0)
		//	ADD	$sz, Rarg0
		//	ADD	$sz, Rarg1
		//	BGEU	Rarg2, Rarg0, -4(PC)

		p := s.Prog(mov)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = riscv.REG_T2

		p2 := s.Prog(mov)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = riscv.REG_T2
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = v.Args[0].Reg()

		p3 := s.Prog(riscv.AADD)
		p3.From.Type = obj.TYPE_CONST
		p3.From.Offset = sz
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = v.Args[0].Reg()

		p4 := s.Prog(riscv.AADD)
		p4.From.Type = obj.TYPE_CONST
		p4.From.Offset = sz
		p4.To.Type = obj.TYPE_REG
		p4.To.Reg = v.Args[1].Reg()

		p5 := s.Prog(riscv.ABGEU)
		p5.To.Type = obj.TYPE_BRANCH
		p5.Reg = v.Args[1].Reg()
		p5.From.Type = obj.TYPE_REG
		p5.From.Reg = v.Args[2].Reg()
		p5.To.SetTarget(p)

	case ssa.OpRISCV64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		// TODO: optimizations. See arm and amd64 LoweredNilCheck.
		p := s.Prog(riscv.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = riscv.REG_ZERO
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos == 1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}

	case ssa.OpRISCV64LoweredGetClosurePtr:
		// Closure pointer is S10 (riscv.REG_CTXT).
		ssagen.CheckLoweredGetClosurePtr(v)

	case ssa.OpRISCV64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(riscv.AMOV)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpRISCV64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

	case ssa.OpRISCV64DUFFZERO:
		p := s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt

	case ssa.OpRISCV64DUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt

	case ssa.OpRISCV64LoweredPubBarrier:
		// FENCE
		s.Prog(v.Op.Asm())

	case ssa.OpRISCV64LoweredRound32F, ssa.OpRISCV64LoweredRound64F:
		// input is already rounded

	case ssa.OpClobber, ssa.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.

	default:
		v.Fatalf("Unhandled op %v", v.Op)
	}
}

var blockBranch = [...]obj.As{
	ssa.BlockRISCV64BEQ:  riscv.ABEQ,
	ssa.BlockRISCV64BEQZ: riscv.ABEQZ,
	ssa.BlockRISCV64BGE:  riscv.ABGE,
	ssa.BlockRISCV64BGEU: riscv.ABGEU,
	ssa.BlockRISCV64BGEZ: riscv.ABGEZ,
	ssa.BlockRISCV64BGTZ: riscv.ABGTZ,
	ssa.BlockRISCV64BLEZ: riscv.ABLEZ,
	ssa.BlockRISCV64BLT:  riscv.ABLT,
	ssa.BlockRISCV64BLTU: riscv.ABLTU,
	ssa.BlockRISCV64BLTZ: riscv.ABLTZ,
	ssa.BlockRISCV64BNE:  riscv.ABNE,
	ssa.BlockRISCV64BNEZ: riscv.ABNEZ,
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	s.SetPos(b.Pos)

	switch b.Kind {
	case ssa.BlockDefer:
		// defer returns in A0:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p := s.Prog(riscv.ABNE)
		p.To.Type = obj.TYPE_BRANCH
		p.From.Type = obj.TYPE_REG
		p.From.Reg = riscv.REG_ZERO
		p.Reg = riscv.REG_A0
		s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[1].Block()})
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockPlain:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
	case ssa.BlockExit, ssa.BlockRetJmp:
	case ssa.BlockRet:
		s.Prog(obj.ARET)
	case ssa.BlockRISCV64BEQ, ssa.BlockRISCV64BEQZ, ssa.BlockRISCV64BNE, ssa.BlockRISCV64BNEZ,
		ssa.BlockRISCV64BLT, ssa.BlockRISCV64BLEZ, ssa.BlockRISCV64BGE, ssa.BlockRISCV64BGEZ,
		ssa.BlockRISCV64BLTZ, ssa.BlockRISCV64BGTZ, ssa.BlockRISCV64BLTU, ssa.BlockRISCV64BGEU:

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
		case ssa.BlockRISCV64BEQ, ssa.BlockRISCV64BNE, ssa.BlockRISCV64BLT, ssa.BlockRISCV64BGE, ssa.BlockRISCV64BLTU, ssa.BlockRISCV64BGEU:
			if b.NumControls() != 2 {
				b.Fatalf("Unexpected number of controls (%d != 2): %s", b.NumControls(), b.LongString())
			}
			p.From.Reg = b.Controls[0].Reg()
			p.Reg = b.Controls[1].Reg()

		case ssa.BlockRISCV64BEQZ, ssa.BlockRISCV64BNEZ, ssa.BlockRISCV64BGEZ, ssa.BlockRISCV64BLEZ, ssa.BlockRISCV64BLTZ, ssa.BlockRISCV64BGTZ:
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
