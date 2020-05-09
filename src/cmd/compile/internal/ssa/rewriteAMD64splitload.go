// Code generated from gen/AMD64splitload.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

func rewriteValueAMD64splitload(v *Value) bool {
	switch v.Op {
	case OpAMD64CMPBconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPBconstload(v)
	case OpAMD64CMPBconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPBconstloadidx1(v)
	case OpAMD64CMPBload:
		return rewriteValueAMD64splitload_OpAMD64CMPBload(v)
	case OpAMD64CMPBloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPBloadidx1(v)
	case OpAMD64CMPLconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstload(v)
	case OpAMD64CMPLconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx1(v)
	case OpAMD64CMPLconstloadidx4:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx4(v)
	case OpAMD64CMPLload:
		return rewriteValueAMD64splitload_OpAMD64CMPLload(v)
	case OpAMD64CMPLloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPLloadidx1(v)
	case OpAMD64CMPLloadidx4:
		return rewriteValueAMD64splitload_OpAMD64CMPLloadidx4(v)
	case OpAMD64CMPQconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstload(v)
	case OpAMD64CMPQconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx1(v)
	case OpAMD64CMPQconstloadidx8:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx8(v)
	case OpAMD64CMPQload:
		return rewriteValueAMD64splitload_OpAMD64CMPQload(v)
	case OpAMD64CMPQloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPQloadidx1(v)
	case OpAMD64CMPQloadidx8:
		return rewriteValueAMD64splitload_OpAMD64CMPQloadidx8(v)
	case OpAMD64CMPWconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstload(v)
	case OpAMD64CMPWconstloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx1(v)
	case OpAMD64CMPWconstloadidx2:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx2(v)
	case OpAMD64CMPWload:
		return rewriteValueAMD64splitload_OpAMD64CMPWload(v)
	case OpAMD64CMPWloadidx1:
		return rewriteValueAMD64splitload_OpAMD64CMPWloadidx1(v)
	case OpAMD64CMPWloadidx2:
		return rewriteValueAMD64splitload_OpAMD64CMPWloadidx2(v)
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTB x:(MOVBload {sym} [vo.Off32()] ptr mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTB)
		x := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPBconst (MOVBload {sym} [vo.Off32()] ptr mem) [vo.Val8()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPBconst)
		v.AuxInt = int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBconstloadidx1(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTB x:(MOVBloadidx1 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTB)
		x := b.NewValue0(v.Pos, OpAMD64MOVBloadidx1, typ.UInt8)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPBconst (MOVBloadidx1 {sym} [vo.Off32()] ptr idx mem) [vo.Val8()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPBconst)
		v.AuxInt = int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBloadidx1, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBload {sym} [off] ptr x mem)
	// result: (CMPB (MOVBload {sym} [off] ptr mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.reset(OpAMD64CMPB)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPBloadidx1(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPB (MOVBloadidx1 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPB)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBloadidx1, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLload {sym} [vo.Off32()] ptr mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTL)
		x := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLload {sym} [vo.Off32()] ptr mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx1(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLloadidx1 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTL)
		x := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLloadidx1 {sym} [vo.Off32()] ptr idx mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstloadidx4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstloadidx4 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTL x:(MOVLloadidx4 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTL)
		x := b.NewValue0(v.Pos, OpAMD64MOVLloadidx4, typ.UInt32)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconstloadidx4 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPLconst (MOVLloadidx4 {sym} [vo.Off32()] ptr idx mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx4, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLload {sym} [off] ptr x mem)
	// result: (CMPL (MOVLload {sym} [off] ptr mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.reset(OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLloadidx1(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPL (MOVLloadidx1 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLloadidx4(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLloadidx4 {sym} [off] ptr idx x mem)
	// result: (CMPL (MOVLloadidx4 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx4, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQload {sym} [vo.Off32()] ptr mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQload {sym} [vo.Off32()] ptr mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx1(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQloadidx1 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQloadidx1 {sym} [vo.Off32()] ptr idx mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstloadidx8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstloadidx8 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTQ x:(MOVQloadidx8 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTQ)
		x := b.NewValue0(v.Pos, OpAMD64MOVQloadidx8, typ.UInt64)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconstloadidx8 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPQconst (MOVQloadidx8 {sym} [vo.Off32()] ptr idx mem) [vo.Val32()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = int32ToAuxInt(vo.Val32())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx8, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQload {sym} [off] ptr x mem)
	// result: (CMPQ (MOVQload {sym} [off] ptr mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.reset(OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQloadidx1(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPQ (MOVQloadidx1 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQloadidx8(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQloadidx8 {sym} [off] ptr idx x mem)
	// result: (CMPQ (MOVQloadidx8 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx8, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWload {sym} [vo.Off32()] ptr mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTW)
		x := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg2(ptr, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWload {sym} [vo.Off32()] ptr mem) [vo.Val16()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx1(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWloadidx1 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTW)
		x := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstloadidx1 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWloadidx1 {sym} [vo.Off32()] ptr idx mem) [vo.Val16()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstloadidx2(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstloadidx2 {sym} [vo] ptr idx mem)
	// cond: vo.Val() == 0
	// result: (TESTW x:(MOVWloadidx2 {sym} [vo.Off32()] ptr idx mem) x)
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() == 0) {
			break
		}
		v.reset(OpAMD64TESTW)
		x := b.NewValue0(v.Pos, OpAMD64MOVWloadidx2, typ.UInt16)
		x.AuxInt = int32ToAuxInt(vo.Off32())
		x.Aux = symToAux(sym)
		x.AddArg3(ptr, idx, mem)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconstloadidx2 {sym} [vo] ptr idx mem)
	// cond: vo.Val() != 0
	// result: (CMPWconst (MOVWloadidx2 {sym} [vo.Off32()] ptr idx mem) [vo.Val16()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		mem := v_2
		if !(vo.Val() != 0) {
			break
		}
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx2, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(vo.Off32())
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPWload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWload {sym} [off] ptr x mem)
	// result: (CMPW (MOVWload {sym} [off] ptr mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		mem := v_2
		v.reset(OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWloadidx1(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWloadidx1 {sym} [off] ptr idx x mem)
	// result: (CMPW (MOVWloadidx1 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWloadidx2(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWloadidx2 {sym} [off] ptr idx x mem)
	// result: (CMPW (MOVWloadidx2 {sym} [off] ptr idx mem) x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		idx := v_1
		x := v_2
		mem := v_3
		v.reset(OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx2, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, idx, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteBlockAMD64splitload(b *Block) bool {
	switch b.Kind {
	}
	return false
}
