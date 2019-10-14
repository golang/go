// Code generated from gen/AMD64splitload.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

func rewriteValueAMD64splitload(v *Value) bool {
	switch v.Op {
	case OpAMD64CMPBconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPBconstload_0(v)
	case OpAMD64CMPBload:
		return rewriteValueAMD64splitload_OpAMD64CMPBload_0(v)
	case OpAMD64CMPLconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPLconstload_0(v)
	case OpAMD64CMPLload:
		return rewriteValueAMD64splitload_OpAMD64CMPLload_0(v)
	case OpAMD64CMPQconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPQconstload_0(v)
	case OpAMD64CMPQload:
		return rewriteValueAMD64splitload_OpAMD64CMPQload_0(v)
	case OpAMD64CMPWconstload:
		return rewriteValueAMD64splitload_OpAMD64CMPWconstload_0(v)
	case OpAMD64CMPWload:
		return rewriteValueAMD64splitload_OpAMD64CMPWload_0(v)
	}
	return false
}
func rewriteValueAMD64splitload_OpAMD64CMPBconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// result: (CMPBconst (MOVBload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpAMD64CMPBconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPBload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBload {sym} [off] ptr x mem)
	// result: (CMPB (MOVBload {sym} [off] ptr mem) x)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64CMPB)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// result: (CMPLconst (MOVLload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPLload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLload {sym} [off] ptr x mem)
	// result: (CMPL (MOVLload {sym} [off] ptr mem) x)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64CMPL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQconstload {sym} [vo] ptr mem)
	// result: (CMPQconst (MOVQload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPQload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPQload {sym} [off] ptr x mem)
	// result: (CMPQ (MOVQload {sym} [off] ptr mem) x)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64CMPQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// result: (CMPWconst (MOVWload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64splitload_OpAMD64CMPWload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWload {sym} [off] ptr x mem)
	// result: (CMPW (MOVWload {sym} [off] ptr mem) x)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64CMPW)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteBlockAMD64splitload(b *Block) bool {
	switch b.Kind {
	}
	return false
}
