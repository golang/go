// Code generated from _gen/386splitload.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValue386splitload(v *Value) bool {
	switch v.Op {
	case Op386CMPBconstload:
		return rewriteValue386splitload_Op386CMPBconstload(v)
	case Op386CMPBload:
		return rewriteValue386splitload_Op386CMPBload(v)
	case Op386CMPLconstload:
		return rewriteValue386splitload_Op386CMPLconstload(v)
	case Op386CMPLload:
		return rewriteValue386splitload_Op386CMPLload(v)
	case Op386CMPWconstload:
		return rewriteValue386splitload_Op386CMPWconstload(v)
	case Op386CMPWload:
		return rewriteValue386splitload_Op386CMPWload(v)
	}
	return false
}
func rewriteValue386splitload_Op386CMPBconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// result: (CMPBconst (MOVBload {sym} [vo.Off()] ptr mem) [vo.Val8()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.reset(Op386CMPBconst)
		v.AuxInt = int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(vo.Off())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPBload(v *Value) bool {
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
		v.reset(Op386CMPB)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// result: (CMPLconst (MOVLload {sym} [vo.Off()] ptr mem) [vo.Val()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.reset(Op386CMPLconst)
		v.AuxInt = int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(vo.Off())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLload(v *Value) bool {
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
		v.reset(Op386CMPL)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// result: (CMPWconst (MOVWload {sym} [vo.Off()] ptr mem) [vo.Val16()])
	for {
		vo := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.reset(Op386CMPWconst)
		v.AuxInt = int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(vo.Off())
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWload(v *Value) bool {
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
		v.reset(Op386CMPW)
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteBlock386splitload(b *Block) bool {
	return false
}
