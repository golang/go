// Code generated from _gen/386splitload.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValue386splitload(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.Op386CMPBconstload:
		return rewriteValue386splitload_Op386CMPBconstload(v)
	case ssaop.Op386CMPBload:
		return rewriteValue386splitload_Op386CMPBload(v)
	case ssaop.Op386CMPLconstload:
		return rewriteValue386splitload_Op386CMPLconstload(v)
	case ssaop.Op386CMPLload:
		return rewriteValue386splitload_Op386CMPLload(v)
	case ssaop.Op386CMPWconstload:
		return rewriteValue386splitload_Op386CMPWconstload(v)
	case ssaop.Op386CMPWload:
		return rewriteValue386splitload_Op386CMPWload(v)
	}
	return false
}
func rewriteValue386splitload_Op386CMPBconstload(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// result: (CMPBconst (MOVBload {sym} [vo.Off()] ptr mem) [vo.Val8()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386CMPBconst)
		v.AuxInt = Int8ToAuxInt(vo.Val8())
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVBload, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPBload(v *ssacore.Value) bool {
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
		v.Reset(ssaop.Op386CMPB)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVBload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLconstload(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// result: (CMPLconst (MOVLload {sym} [vo.Off()] ptr mem) [vo.Val()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386CMPLconst)
		v.AuxInt = Int32ToAuxInt(vo.Val())
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLload(v *ssacore.Value) bool {
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
		v.Reset(ssaop.Op386CMPL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWconstload(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// result: (CMPWconst (MOVWload {sym} [vo.Off()] ptr mem) [vo.Val16()])
	for {
		vo := AuxIntToValAndOff(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386CMPWconst)
		v.AuxInt = Int16ToAuxInt(vo.Val16())
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVWload, typ.UInt16)
		v0.AuxInt = Int32ToAuxInt(vo.Off())
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWload(v *ssacore.Value) bool {
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
		v.Reset(ssaop.Op386CMPW)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVWload, x.Type)
		v0.AuxInt = Int32ToAuxInt(off)
		v0.Aux = SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteBlock386splitload(b *ssacore.Block) bool {
	return false
}
