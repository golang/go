// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/compile/internal/gc"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"
	"cmd/internal/objabi"
)

func Init(arch *gc.Arch) {
	arch.LinkArch = &wasm.Linkwasm
	arch.REGSP = wasm.REG_SP
	arch.MAXWIDTH = 1 << 50

	arch.ZeroRange = zeroRange
	arch.Ginsnop = ginsnop
	arch.Ginsnopdefer = ginsnop

	arch.SSAMarkMoves = ssaMarkMoves
	arch.SSAGenValue = ssaGenValue
	arch.SSAGenBlock = ssaGenBlock
}

func zeroRange(pp *gc.Progs, p *obj.Prog, off, cnt int64, state *uint32) *obj.Prog {
	if cnt == 0 {
		return p
	}
	if cnt%8 != 0 {
		gc.Fatalf("zerorange count not a multiple of widthptr %d", cnt)
	}

	for i := int64(0); i < cnt; i += 8 {
		p = pp.Appendpp(p, wasm.AGet, obj.TYPE_REG, wasm.REG_SP, 0, 0, 0, 0)
		p = pp.Appendpp(p, wasm.AI64Const, obj.TYPE_CONST, 0, 0, 0, 0, 0)
		p = pp.Appendpp(p, wasm.AI64Store, 0, 0, 0, obj.TYPE_CONST, 0, off+i)
	}

	return p
}

func ginsnop(pp *gc.Progs) *obj.Prog {
	return pp.Prog(wasm.ANop)
}

func ssaMarkMoves(s *gc.SSAGenState, b *ssa.Block) {
}

func ssaGenBlock(s *gc.SSAGenState, b, next *ssa.Block) {
	switch b.Kind {
	case ssa.BlockPlain:
		if next != b.Succs[0].Block() {
			s.Br(obj.AJMP, b.Succs[0].Block())
		}

	case ssa.BlockIf:
		switch next {
		case b.Succs[0].Block():
			// if false, jump to b.Succs[1]
			getValue32(s, b.Controls[0])
			s.Prog(wasm.AI32Eqz)
			s.Prog(wasm.AIf)
			s.Br(obj.AJMP, b.Succs[1].Block())
			s.Prog(wasm.AEnd)
		case b.Succs[1].Block():
			// if true, jump to b.Succs[0]
			getValue32(s, b.Controls[0])
			s.Prog(wasm.AIf)
			s.Br(obj.AJMP, b.Succs[0].Block())
			s.Prog(wasm.AEnd)
		default:
			// if true, jump to b.Succs[0], else jump to b.Succs[1]
			getValue32(s, b.Controls[0])
			s.Prog(wasm.AIf)
			s.Br(obj.AJMP, b.Succs[0].Block())
			s.Prog(wasm.AEnd)
			s.Br(obj.AJMP, b.Succs[1].Block())
		}

	case ssa.BlockRet:
		s.Prog(obj.ARET)

	case ssa.BlockRetJmp:
		p := s.Prog(obj.ARET)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = b.Aux.(*obj.LSym)

	case ssa.BlockExit:

	case ssa.BlockDefer:
		p := s.Prog(wasm.AGet)
		p.From = obj.Addr{Type: obj.TYPE_REG, Reg: wasm.REG_RET0}
		s.Prog(wasm.AI64Eqz)
		s.Prog(wasm.AI32Eqz)
		s.Prog(wasm.AIf)
		s.Br(obj.AJMP, b.Succs[1].Block())
		s.Prog(wasm.AEnd)
		if next != b.Succs[0].Block() {
			s.Br(obj.AJMP, b.Succs[0].Block())
		}

	default:
		panic("unexpected block")
	}

	// Entry point for the next block. Used by the JMP in goToBlock.
	s.Prog(wasm.ARESUMEPOINT)

	if s.OnWasmStackSkipped != 0 {
		panic("wasm: bad stack")
	}
}

func ssaGenValue(s *gc.SSAGenState, v *ssa.Value) {
	switch v.Op {
	case ssa.OpWasmLoweredStaticCall, ssa.OpWasmLoweredClosureCall, ssa.OpWasmLoweredInterCall:
		s.PrepareCall(v)
		if call, ok := v.Aux.(*ssa.AuxCall); ok && call.Fn == gc.Deferreturn {
			// add a resume point before call to deferreturn so it can be called again via jmpdefer
			s.Prog(wasm.ARESUMEPOINT)
		}
		if v.Op == ssa.OpWasmLoweredClosureCall {
			getValue64(s, v.Args[1])
			setReg(s, wasm.REG_CTXT)
		}
		if call, ok := v.Aux.(*ssa.AuxCall); ok && call.Fn != nil {
			sym := call.Fn
			p := s.Prog(obj.ACALL)
			p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: sym}
			p.Pos = v.Pos
		} else {
			getValue64(s, v.Args[0])
			p := s.Prog(obj.ACALL)
			p.To = obj.Addr{Type: obj.TYPE_NONE}
			p.Pos = v.Pos
		}

	case ssa.OpWasmLoweredMove:
		getValue32(s, v.Args[0])
		getValue32(s, v.Args[1])
		i32Const(s, int32(v.AuxInt))
		p := s.Prog(wasm.ACall)
		p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.WasmMove}

	case ssa.OpWasmLoweredZero:
		getValue32(s, v.Args[0])
		i32Const(s, int32(v.AuxInt))
		p := s.Prog(wasm.ACall)
		p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.WasmZero}

	case ssa.OpWasmLoweredNilCheck:
		getValue64(s, v.Args[0])
		s.Prog(wasm.AI64Eqz)
		s.Prog(wasm.AIf)
		p := s.Prog(wasm.ACALLNORESUME)
		p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.SigPanic}
		s.Prog(wasm.AEnd)
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if gc.Debug_checknil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			gc.Warnl(v.Pos, "generated nil check")
		}

	case ssa.OpWasmLoweredWB:
		getValue64(s, v.Args[0])
		getValue64(s, v.Args[1])
		p := s.Prog(wasm.ACALLNORESUME) // TODO(neelance): If possible, turn this into a simple wasm.ACall).
		p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: v.Aux.(*obj.LSym)}

	case ssa.OpWasmI64Store8, ssa.OpWasmI64Store16, ssa.OpWasmI64Store32, ssa.OpWasmI64Store, ssa.OpWasmF32Store, ssa.OpWasmF64Store:
		getValue32(s, v.Args[0])
		getValue64(s, v.Args[1])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}

	case ssa.OpStoreReg:
		getReg(s, wasm.REG_SP)
		getValue64(s, v.Args[0])
		p := s.Prog(storeOp(v.Type))
		gc.AddrAuto(&p.To, v)

	default:
		if v.Type.IsMemory() {
			return
		}
		if v.OnWasmStack {
			s.OnWasmStackSkipped++
			// If a Value is marked OnWasmStack, we don't generate the value and store it to a register now.
			// Instead, we delay the generation to when the value is used and then directly generate it on the WebAssembly stack.
			return
		}
		ssaGenValueOnStack(s, v, true)
		if s.OnWasmStackSkipped != 0 {
			panic("wasm: bad stack")
		}
		setReg(s, v.Reg())
	}
}

func ssaGenValueOnStack(s *gc.SSAGenState, v *ssa.Value, extend bool) {
	switch v.Op {
	case ssa.OpWasmLoweredGetClosurePtr:
		getReg(s, wasm.REG_CTXT)

	case ssa.OpWasmLoweredGetCallerPC:
		p := s.Prog(wasm.AI64Load)
		// Caller PC is stored 8 bytes below first parameter.
		p.From = obj.Addr{
			Type:   obj.TYPE_MEM,
			Name:   obj.NAME_PARAM,
			Offset: -8,
		}

	case ssa.OpWasmLoweredGetCallerSP:
		p := s.Prog(wasm.AGet)
		// Caller SP is the address of the first parameter.
		p.From = obj.Addr{
			Type:   obj.TYPE_ADDR,
			Name:   obj.NAME_PARAM,
			Reg:    wasm.REG_SP,
			Offset: 0,
		}

	case ssa.OpWasmLoweredAddr:
		if v.Aux == nil { // address of off(SP), no symbol
			getValue64(s, v.Args[0])
			i64Const(s, v.AuxInt)
			s.Prog(wasm.AI64Add)
			break
		}
		p := s.Prog(wasm.AGet)
		p.From.Type = obj.TYPE_ADDR
		switch v.Aux.(type) {
		case *obj.LSym:
			gc.AddAux(&p.From, v)
		case *gc.Node:
			p.From.Reg = v.Args[0].Reg()
			gc.AddAux(&p.From, v)
		default:
			panic("wasm: bad LoweredAddr")
		}

	case ssa.OpWasmLoweredConvert:
		getValue64(s, v.Args[0])

	case ssa.OpWasmSelect:
		getValue64(s, v.Args[0])
		getValue64(s, v.Args[1])
		getValue32(s, v.Args[2])
		s.Prog(v.Op.Asm())

	case ssa.OpWasmI64AddConst:
		getValue64(s, v.Args[0])
		i64Const(s, v.AuxInt)
		s.Prog(v.Op.Asm())

	case ssa.OpWasmI64Const:
		i64Const(s, v.AuxInt)

	case ssa.OpWasmF32Const:
		f32Const(s, v.AuxFloat())

	case ssa.OpWasmF64Const:
		f64Const(s, v.AuxFloat())

	case ssa.OpWasmI64Load8U, ssa.OpWasmI64Load8S, ssa.OpWasmI64Load16U, ssa.OpWasmI64Load16S, ssa.OpWasmI64Load32U, ssa.OpWasmI64Load32S, ssa.OpWasmI64Load, ssa.OpWasmF32Load, ssa.OpWasmF64Load:
		getValue32(s, v.Args[0])
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}

	case ssa.OpWasmI64Eqz:
		getValue64(s, v.Args[0])
		s.Prog(v.Op.Asm())
		if extend {
			s.Prog(wasm.AI64ExtendI32U)
		}

	case ssa.OpWasmI64Eq, ssa.OpWasmI64Ne, ssa.OpWasmI64LtS, ssa.OpWasmI64LtU, ssa.OpWasmI64GtS, ssa.OpWasmI64GtU, ssa.OpWasmI64LeS, ssa.OpWasmI64LeU, ssa.OpWasmI64GeS, ssa.OpWasmI64GeU,
		ssa.OpWasmF32Eq, ssa.OpWasmF32Ne, ssa.OpWasmF32Lt, ssa.OpWasmF32Gt, ssa.OpWasmF32Le, ssa.OpWasmF32Ge,
		ssa.OpWasmF64Eq, ssa.OpWasmF64Ne, ssa.OpWasmF64Lt, ssa.OpWasmF64Gt, ssa.OpWasmF64Le, ssa.OpWasmF64Ge:
		getValue64(s, v.Args[0])
		getValue64(s, v.Args[1])
		s.Prog(v.Op.Asm())
		if extend {
			s.Prog(wasm.AI64ExtendI32U)
		}

	case ssa.OpWasmI64Add, ssa.OpWasmI64Sub, ssa.OpWasmI64Mul, ssa.OpWasmI64DivU, ssa.OpWasmI64RemS, ssa.OpWasmI64RemU, ssa.OpWasmI64And, ssa.OpWasmI64Or, ssa.OpWasmI64Xor, ssa.OpWasmI64Shl, ssa.OpWasmI64ShrS, ssa.OpWasmI64ShrU, ssa.OpWasmI64Rotl,
		ssa.OpWasmF32Add, ssa.OpWasmF32Sub, ssa.OpWasmF32Mul, ssa.OpWasmF32Div, ssa.OpWasmF32Copysign,
		ssa.OpWasmF64Add, ssa.OpWasmF64Sub, ssa.OpWasmF64Mul, ssa.OpWasmF64Div, ssa.OpWasmF64Copysign:
		getValue64(s, v.Args[0])
		getValue64(s, v.Args[1])
		s.Prog(v.Op.Asm())

	case ssa.OpWasmI32Rotl:
		getValue32(s, v.Args[0])
		getValue32(s, v.Args[1])
		s.Prog(wasm.AI32Rotl)
		s.Prog(wasm.AI64ExtendI32U)

	case ssa.OpWasmI64DivS:
		getValue64(s, v.Args[0])
		getValue64(s, v.Args[1])
		if v.Type.Size() == 8 {
			// Division of int64 needs helper function wasmDiv to handle the MinInt64 / -1 case.
			p := s.Prog(wasm.ACall)
			p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.WasmDiv}
			break
		}
		s.Prog(wasm.AI64DivS)

	case ssa.OpWasmI64TruncSatF32S, ssa.OpWasmI64TruncSatF64S:
		getValue64(s, v.Args[0])
		if objabi.GOWASM.SatConv {
			s.Prog(v.Op.Asm())
		} else {
			if v.Op == ssa.OpWasmI64TruncSatF32S {
				s.Prog(wasm.AF64PromoteF32)
			}
			p := s.Prog(wasm.ACall)
			p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.WasmTruncS}
		}

	case ssa.OpWasmI64TruncSatF32U, ssa.OpWasmI64TruncSatF64U:
		getValue64(s, v.Args[0])
		if objabi.GOWASM.SatConv {
			s.Prog(v.Op.Asm())
		} else {
			if v.Op == ssa.OpWasmI64TruncSatF32U {
				s.Prog(wasm.AF64PromoteF32)
			}
			p := s.Prog(wasm.ACall)
			p.To = obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: gc.WasmTruncU}
		}

	case ssa.OpWasmF32DemoteF64:
		getValue64(s, v.Args[0])
		s.Prog(v.Op.Asm())

	case ssa.OpWasmF64PromoteF32:
		getValue64(s, v.Args[0])
		s.Prog(v.Op.Asm())

	case ssa.OpWasmF32ConvertI64S, ssa.OpWasmF32ConvertI64U,
		ssa.OpWasmF64ConvertI64S, ssa.OpWasmF64ConvertI64U,
		ssa.OpWasmI64Extend8S, ssa.OpWasmI64Extend16S, ssa.OpWasmI64Extend32S,
		ssa.OpWasmF32Neg, ssa.OpWasmF32Sqrt, ssa.OpWasmF32Trunc, ssa.OpWasmF32Ceil, ssa.OpWasmF32Floor, ssa.OpWasmF32Nearest, ssa.OpWasmF32Abs,
		ssa.OpWasmF64Neg, ssa.OpWasmF64Sqrt, ssa.OpWasmF64Trunc, ssa.OpWasmF64Ceil, ssa.OpWasmF64Floor, ssa.OpWasmF64Nearest, ssa.OpWasmF64Abs,
		ssa.OpWasmI64Ctz, ssa.OpWasmI64Clz, ssa.OpWasmI64Popcnt:
		getValue64(s, v.Args[0])
		s.Prog(v.Op.Asm())

	case ssa.OpLoadReg:
		p := s.Prog(loadOp(v.Type))
		gc.AddrAuto(&p.From, v.Args[0])

	case ssa.OpCopy:
		getValue64(s, v.Args[0])

	default:
		v.Fatalf("unexpected op: %s", v.Op)

	}
}

func isCmp(v *ssa.Value) bool {
	switch v.Op {
	case ssa.OpWasmI64Eqz, ssa.OpWasmI64Eq, ssa.OpWasmI64Ne, ssa.OpWasmI64LtS, ssa.OpWasmI64LtU, ssa.OpWasmI64GtS, ssa.OpWasmI64GtU, ssa.OpWasmI64LeS, ssa.OpWasmI64LeU, ssa.OpWasmI64GeS, ssa.OpWasmI64GeU,
		ssa.OpWasmF32Eq, ssa.OpWasmF32Ne, ssa.OpWasmF32Lt, ssa.OpWasmF32Gt, ssa.OpWasmF32Le, ssa.OpWasmF32Ge,
		ssa.OpWasmF64Eq, ssa.OpWasmF64Ne, ssa.OpWasmF64Lt, ssa.OpWasmF64Gt, ssa.OpWasmF64Le, ssa.OpWasmF64Ge:
		return true
	default:
		return false
	}
}

func getValue32(s *gc.SSAGenState, v *ssa.Value) {
	if v.OnWasmStack {
		s.OnWasmStackSkipped--
		ssaGenValueOnStack(s, v, false)
		if !isCmp(v) {
			s.Prog(wasm.AI32WrapI64)
		}
		return
	}

	reg := v.Reg()
	getReg(s, reg)
	if reg != wasm.REG_SP {
		s.Prog(wasm.AI32WrapI64)
	}
}

func getValue64(s *gc.SSAGenState, v *ssa.Value) {
	if v.OnWasmStack {
		s.OnWasmStackSkipped--
		ssaGenValueOnStack(s, v, true)
		return
	}

	reg := v.Reg()
	getReg(s, reg)
	if reg == wasm.REG_SP {
		s.Prog(wasm.AI64ExtendI32U)
	}
}

func i32Const(s *gc.SSAGenState, val int32) {
	p := s.Prog(wasm.AI32Const)
	p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(val)}
}

func i64Const(s *gc.SSAGenState, val int64) {
	p := s.Prog(wasm.AI64Const)
	p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: val}
}

func f32Const(s *gc.SSAGenState, val float64) {
	p := s.Prog(wasm.AF32Const)
	p.From = obj.Addr{Type: obj.TYPE_FCONST, Val: val}
}

func f64Const(s *gc.SSAGenState, val float64) {
	p := s.Prog(wasm.AF64Const)
	p.From = obj.Addr{Type: obj.TYPE_FCONST, Val: val}
}

func getReg(s *gc.SSAGenState, reg int16) {
	p := s.Prog(wasm.AGet)
	p.From = obj.Addr{Type: obj.TYPE_REG, Reg: reg}
}

func setReg(s *gc.SSAGenState, reg int16) {
	p := s.Prog(wasm.ASet)
	p.To = obj.Addr{Type: obj.TYPE_REG, Reg: reg}
}

func loadOp(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return wasm.AF32Load
		case 8:
			return wasm.AF64Load
		default:
			panic("bad load type")
		}
	}

	switch t.Size() {
	case 1:
		if t.IsSigned() {
			return wasm.AI64Load8S
		}
		return wasm.AI64Load8U
	case 2:
		if t.IsSigned() {
			return wasm.AI64Load16S
		}
		return wasm.AI64Load16U
	case 4:
		if t.IsSigned() {
			return wasm.AI64Load32S
		}
		return wasm.AI64Load32U
	case 8:
		return wasm.AI64Load
	default:
		panic("bad load type")
	}
}

func storeOp(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return wasm.AF32Store
		case 8:
			return wasm.AF64Store
		default:
			panic("bad store type")
		}
	}

	switch t.Size() {
	case 1:
		return wasm.AI64Store8
	case 2:
		return wasm.AI64Store16
	case 4:
		return wasm.AI64Store32
	case 8:
		return wasm.AI64Store
	default:
		panic("bad store type")
	}
}
