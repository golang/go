// Code generated from _gen/AMD64splitload.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValueAMD64splitload(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAMD64CMPBconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPBconstload(v)
	case ssaop.OpAMD64CMPBconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPBconstloadidx1(v)
	case ssaop.OpAMD64CMPBload:
		return rewriteValueAMD64splitload_OpAMD64CMPBload(v)
	case ssaop.OpAMD64CMPBloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPBloadidx1(v)
	case ssaop.OpAMD64CMPLconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstload(v)
	case ssaop.OpAMD64CMPLconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx1(v)
	case ssaop.OpAMD64CMPLconstloadidx4:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx4(v)
	case ssaop.OpAMD64CMPLload:
		return rewriteValueAMD64splitload_OpAMD64CMPLload(v)
	case ssaop.OpAMD64CMPLloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPLloadidx1(v)
	case ssaop.OpAMD64CMPLloadidx4:
		return rewriteValueAMD64splitload_OpAMD64CMPLloadidx4(v)
	case ssaop.OpAMD64CMPQconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstload(v)
	case ssaop.OpAMD64CMPQconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx1(v)
	case ssaop.OpAMD64CMPQconstloadidx8:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx8(v)
	case ssaop.OpAMD64CMPQload:
		return rewriteValueAMD64splitload_OpAMD64CMPQload(v)
	case ssaop.OpAMD64CMPQloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPQloadidx1(v)
	case ssaop.OpAMD64CMPQloadidx8:
		return rewriteValueAMD64splitload_OpAMD64CMPQloadidx8(v)
	case ssaop.OpAMD64CMPWconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstload(v)
	case ssaop.OpAMD64CMPWconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx1(v)
	case ssaop.OpAMD64CMPWconstloadidx2:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx2(v)
	case ssaop.OpAMD64CMPWload:
		return rewriteValueAMD64splitload_OpAMD64CMPWload(v)
	case ssaop.OpAMD64CMPWloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPWloadidx1(v)
	case ssaop.OpAMD64CMPWloadidx2:
		return rewriteValueAMD64splitload_OpAMD64CMPWloadidx2(v)
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBconstload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTB x:(MOVBload {sym} [vo.Off()] ptr mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTB)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBload, typ.UInt8)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPBconst (MOVBload {sym} [vo.Off()] ptr mem) [vo.Val8()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPBconst)
		v.AuxInt = Int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBconstloadidx1(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTB x:(MOVBloadidx1 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTB)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBloadidx1, typ.UInt8)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPBconst (MOVBloadidx1 {sym} [vo.Off()] ptr idx mem) [vo.Val8()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPBconst)
		v.AuxInt = Int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBloadidx1, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPBload {sym} [off] ptr x mem)
	// result: (CMPB (MOVBload <x.Type> {sym} [off] ptr mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.Reset(ssaop.OpAMD64CMPB)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPBloadidx1(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPBloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPB (MOVBloadidx1 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPB)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVBloadidx1, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLload {sym} [vo.Off()] ptr mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTL)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLload, typ.UInt32)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLload {sym} [vo.Off()] ptr mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPLconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx1(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLloadidx1 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTL)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx1, typ.UInt32)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLloadidx1 {sym} [vo.Off()] ptr idx mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPLconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx1, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstloadidx4 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLloadidx4 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTL)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx4, typ.UInt32)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstloadidx4 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLloadidx4 {sym} [vo.Off()] ptr idx mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPLconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx4, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLload {sym} [off] ptr x mem)
	// result: (CMPL (MOVLload <x.Type> {sym} [off] ptr mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.Reset(ssaop.OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLloadidx1(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPL (MOVLloadidx1 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx1, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLloadidx4(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLloadidx4 {sym} [off] ptr idx x mem)
	// result: (CMPL (MOVLloadidx4 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVLloadidx4, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQload {sym} [vo.Off()] ptr mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQload, typ.UInt64)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQload {sym} [vo.Off()] ptr mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPQconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx1(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQloadidx1 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx1, typ.UInt64)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQloadidx1 {sym} [vo.Off()] ptr idx mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPQconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx1, typ.UInt64)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstloadidx8 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQloadidx8 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx8, typ.UInt64)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstloadidx8 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQloadidx8 {sym} [vo.Off()] ptr idx mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPQconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx8, typ.UInt64)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPQload {sym} [off] ptr x mem)
	// result: (CMPQ (MOVQload <x.Type> {sym} [off] ptr mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.Reset(ssaop.OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQloadidx1(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPQloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPQ (MOVQloadidx1 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx1, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQloadidx8(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPQloadidx8 {sym} [off] ptr idx x mem)
	// result: (CMPQ (MOVQloadidx8 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVQloadidx8, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWload {sym} [vo.Off()] ptr mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTW)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWload, typ.UInt16)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWload {sym} [vo.Off()] ptr mem) [vo.Val16()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPWconst)
		v.AuxInt = Int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx1(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWloadidx1 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTW)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx1, typ.UInt16)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWloadidx1 {sym} [vo.Off()] ptr idx mem) [vo.Val16()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPWconst)
		v.AuxInt = Int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx1, typ.UInt16)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx2(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstloadidx2 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWloadidx2 {sym} [vo.Off()] ptr idx mem) x)
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.Reset(ssaop.OpAMD64TESTW)
		x := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx2, typ.UInt16)
		x.AuxInt = Int32ToAuxInt(vo.Off())
		x.Aux = SymToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstloadidx2 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWloadidx2 {sym} [vo.Off()] ptr idx mem) [vo.Val16()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.Reset(ssaop.OpAMD64CMPWconst)
		v.AuxInt = Int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx2, typ.UInt16)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWload {sym} [off] ptr x mem)
	// result: (CMPW (MOVWload <x.Type> {sym} [off] ptr mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.Reset(ssaop.OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWloadidx1(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPW (MOVWloadidx1 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx1, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWloadidx2(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWloadidx2 {sym} [off] ptr idx x mem)
	// result: (CMPW (MOVWloadidx2 <x.Type> {sym} [off] ptr idx mem) x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.Reset(ssaop.OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64MOVWloadidx2, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteBlockAMD64splitload(b *ssa.Block) bool {
	return false
}
