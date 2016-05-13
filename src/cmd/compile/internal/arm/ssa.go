// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

var ssaRegToReg = []int16{
	arm.REG_R0,
	arm.REG_R1,
	arm.REG_R2,
	arm.REG_R3,
	arm.REG_R4,
	arm.REG_R5,
	arm.REG_R6,
	arm.REG_R7,
	arm.REG_R8,
	arm.REG_R9,
	arm.REG_R10,
	arm.REG_R11,
	arm.REG_R12,
	arm.REGSP, // aka R13
	arm.REG_R14,
	arm.REG_R15,

	arm.REG_CPSR, // flag
	0,            // SB isn't a real register.  We fill an Addr.Reg field with 0 in this case.
}

func ssaGenValue(s *gc.SSAGenState, v *ssa.Value) {
	s.SetLineno(v.Line)
	switch v.Op {
	case ssa.OpInitMem:
		// memory arg needs no code
	case ssa.OpArg:
		// input args need no code
	case ssa.OpSP, ssa.OpSB:
		// nothing to do
	case ssa.OpCopy:
		if v.Type.IsMemory() {
			return
		}
		x := gc.SSARegNum(v.Args[0])
		y := gc.SSARegNum(v)
		if x == y {
			return
		}
		p := gc.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssa.OpLoadReg:
		// TODO: by type
		p := gc.Prog(arm.AMOVW)
		n, off := gc.AutoVar(v.Args[0])
		p.From.Type = obj.TYPE_MEM
		p.From.Node = n
		p.From.Sym = gc.Linksym(n.Sym)
		p.From.Offset = off
		if n.Class == gc.PPARAM || n.Class == gc.PPARAMOUT {
			p.From.Name = obj.NAME_PARAM
			p.From.Offset += n.Xoffset
		} else {
			p.From.Name = obj.NAME_AUTO
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	case ssa.OpPhi:
		// just check to make sure regalloc and stackalloc did it right
		if v.Type.IsMemory() {
			return
		}
		f := v.Block.Func
		loc := f.RegAlloc[v.ID]
		for _, a := range v.Args {
			if aloc := f.RegAlloc[a.ID]; aloc != loc { // TODO: .Equal() instead?
				v.Fatalf("phi arg at different location than phi: %v @ %v, but arg %v @ %v\n%s\n", v, loc, a, aloc, v.Block.Func)
			}
		}
	case ssa.OpStoreReg:
		// TODO: by type
		p := gc.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		n, off := gc.AutoVar(v)
		p.To.Type = obj.TYPE_MEM
		p.To.Node = n
		p.To.Sym = gc.Linksym(n.Sym)
		p.To.Offset = off
		if n.Class == gc.PPARAM || n.Class == gc.PPARAMOUT {
			p.To.Name = obj.NAME_PARAM
			p.To.Offset += n.Xoffset
		} else {
			p.To.Name = obj.NAME_AUTO
		}
	case ssa.OpARMADD,
		ssa.OpARMSUB,
		ssa.OpARMRSB,
		ssa.OpARMAND,
		ssa.OpARMOR,
		ssa.OpARMXOR,
		ssa.OpARMBIC,
		ssa.OpARMMUL:
		r := gc.SSARegNum(v)
		r1 := gc.SSARegNum(v.Args[0])
		r2 := gc.SSARegNum(v.Args[1])
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMSLL,
		ssa.OpARMSRL:
		// ARM shift instructions uses only the low-order byte of the shift amount
		// generate conditional instructions to deal with large shifts
		// CMP	$32, Rarg1
		// SLL	Rarg1, Rarg0, Rdst
		// MOVW.HS	$0, Rdst
		r := gc.SSARegNum(v)
		r1 := gc.SSARegNum(v.Args[0])
		r2 := gc.SSARegNum(v.Args[1])
		p := gc.Prog(arm.ACMP)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 32
		p.Reg = r2
		p = gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p = gc.Prog(arm.AMOVW)
		p.Scond = arm.C_SCOND_HS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpARMSRA:
		// ARM shift instructions uses only the low-order byte of the shift amount
		// generate conditional instructions to deal with large shifts
		// CMP	$32, Rarg1
		// SRA.HS	$31, Rarg0, Rdst // shift 31 bits to get the sign bit
		// SRA.LO	Rarg1, Rarg0, Rdst
		r := gc.SSARegNum(v)
		r1 := gc.SSARegNum(v.Args[0])
		r2 := gc.SSARegNum(v.Args[1])
		p := gc.Prog(arm.ACMP)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 32
		p.Reg = r2
		p = gc.Prog(arm.ASRA)
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
	case ssa.OpARMADDconst:
		if v.Aux != nil {
			switch v.Aux.(type) {
			default:
				v.Fatalf("aux is of unknown type %T", v.Aux)
			case *ssa.ExternSymbol:
				reg := v.Args[0].Block.Func.RegAlloc[v.Args[0].ID].(*ssa.Register)
				if reg.Name() != "SB" {
					v.Fatalf("extern symbol with non-SB base register %s", reg.Name())
				}
			case *ssa.ArgSymbol,
				*ssa.AutoSymbol:
				reg := v.Args[0].Block.Func.RegAlloc[v.Args[0].ID].(*ssa.Register)
				if reg.Name() != "SP" {
					v.Fatalf("arg/auto symbol with non-SP base register %s", reg.Name())
				}
			}
			// MOVW $sym+off(base), R
			// the assembler expands it as the following:
			// - base is SP: add constant offset to SP (R13)
			//               when constant is large, tmp register (R11) may be used
			// - base is SB: load external address from constant pool (use relocation)
			p := gc.Prog(arm.AMOVW)
			p.From.Type = obj.TYPE_ADDR
			gc.AddAux(&p.From, v)
			p.To.Type = obj.TYPE_REG
			p.To.Reg = gc.SSARegNum(v)
			break
		}
		fallthrough
	case ssa.OpARMSUBconst,
		ssa.OpARMRSBconst,
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
		p.Reg = gc.SSARegNum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	case ssa.OpARMHMUL,
		ssa.OpARMHMULU:
		// 32-bit high multiplication
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.Reg = gc.SSARegNum(v.Args[1])
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = gc.SSARegNum(v)
		p.To.Offset = arm.REGTMP // throw away low 32-bit into tmp register
	case ssa.OpARMMOVWconst:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	case ssa.OpARMCMP,
		ssa.OpARMCMN,
		ssa.OpARMTST,
		ssa.OpARMTEQ:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		// Special layout in ARM assembly
		// Comparing to x86, the operands of ARM's CMP are reversed.
		p.From.Reg = gc.SSARegNum(v.Args[1])
		p.Reg = gc.SSARegNum(v.Args[0])
	case ssa.OpARMCMPconst,
		ssa.OpARMCMNconst,
		ssa.OpARMTSTconst,
		ssa.OpARMTEQconst:
		// Special layout in ARM assembly
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = gc.SSARegNum(v.Args[0])
	case ssa.OpARMMOVBload,
		ssa.OpARMMOVBUload,
		ssa.OpARMMOVHload,
		ssa.OpARMMOVHUload,
		ssa.OpARMMOVWload:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	case ssa.OpARMMOVBstore,
		ssa.OpARMMOVHstore,
		ssa.OpARMMOVWstore:
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.To, v)
	case ssa.OpARMMOVBreg,
		ssa.OpARMMOVBUreg,
		ssa.OpARMMOVHreg,
		ssa.OpARMMOVHUreg,
		ssa.OpARMMVN:
		if v.Type.IsMemory() {
			v.Fatalf("memory operand for %s", v.LongString())
		}
		p := gc.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	case ssa.OpARMCALLstatic:
		// TODO: deferreturn
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
		p.To.Reg = gc.SSARegNum(v.Args[0])
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
		p.To.Reg = gc.SSARegNum(v.Args[0])
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
		p.From.Reg = gc.SSARegNum(v.Args[0])
		gc.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		if gc.Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			gc.Warnl(v.Line, "generated nil check")
		}
	case ssa.OpARMLoweredZero:
		// MOVW.P	Rarg2, 4(R1)
		// CMP	Rarg1, R1
		// BLT	-2(PC)
		// arg1 is the end of memory to zero
		// arg2 is known to be zero
		p := gc.Prog(arm.AMOVW)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = gc.SSARegNum(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm.REG_R1
		p.To.Offset = 4
		p2 := gc.Prog(arm.ACMP)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = gc.SSARegNum(v.Args[1])
		p2.Reg = arm.REG_R1
		p3 := gc.Prog(arm.ABLT)
		p3.To.Type = obj.TYPE_BRANCH
		gc.Patch(p3, p)
	case ssa.OpARMLoweredMove:
		// MOVW.P	4(R1), Rtmp
		// MOVW.P	Rtmp, 4(R2)
		// CMP	Rarg2, R1
		// BLT	-3(PC)
		// arg2 is the end of src
		p := gc.Prog(arm.AMOVW)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = arm.REG_R1
		p.From.Offset = 4
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		p2 := gc.Prog(arm.AMOVW)
		p2.Scond = arm.C_PBIT
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = arm.REG_R2
		p2.To.Offset = 4
		p3 := gc.Prog(arm.ACMP)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = gc.SSARegNum(v.Args[2])
		p3.Reg = arm.REG_R1
		p4 := gc.Prog(arm.ABLT)
		p4.To.Type = obj.TYPE_BRANCH
		gc.Patch(p4, p)
	case ssa.OpVarDef:
		gc.Gvardef(v.Aux.(*gc.Node))
	case ssa.OpVarKill:
		gc.Gvarkill(v.Aux.(*gc.Node))
	case ssa.OpVarLive:
		gc.Gvarlive(v.Aux.(*gc.Node))
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
		p.To.Reg = gc.SSARegNum(v)
		p = gc.Prog(arm.AMOVW)
		p.Scond = condBits[v.Op]
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = gc.SSARegNum(v)
	default:
		v.Unimplementedf("genValue not implemented: %s", v.LongString())
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
	case ssa.BlockPlain, ssa.BlockCall, ssa.BlockCheck:
		if b.Succs[0].Block() != next {
			p := gc.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, gc.Branch{P: p, B: b.Succs[0].Block()})
		}

	case ssa.BlockExit:
		gc.Prog(obj.AUNDEF) // tell plive.go that we never reach here

	case ssa.BlockRet:
		gc.Prog(obj.ARET)

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
		b.Unimplementedf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}
