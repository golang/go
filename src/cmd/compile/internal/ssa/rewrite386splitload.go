// Code generated from gen/386splitload.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

func rewriteValue386splitload(v *Value) bool {
	switch v.Op {
	case Op386CMPBconstload:
		return rewriteValue386splitload_Op386CMPBconstload_0(v)
	case Op386CMPBload:
		return rewriteValue386splitload_Op386CMPBload_0(v)
	case Op386CMPLconstload:
		return rewriteValue386splitload_Op386CMPLconstload_0(v)
	case Op386CMPLload:
		return rewriteValue386splitload_Op386CMPLload_0(v)
	case Op386CMPWconstload:
		return rewriteValue386splitload_Op386CMPWconstload_0(v)
	case Op386CMPWload:
		return rewriteValue386splitload_Op386CMPWload_0(v)
	}
	return false
}
func rewriteValue386splitload_Op386CMPBconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPBconstload {sym} [vo] ptr mem)
	// result: (CMPBconst (MOVBload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(Op386CMPBconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPBload_0(v *Value) bool {
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
		v.reset(Op386CMPB)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPLconstload {sym} [vo] ptr mem)
	// result: (CMPLconst (MOVLload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(Op386CMPLconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPLload_0(v *Value) bool {
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
		v.reset(Op386CMPL)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWconstload_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconstload {sym} [vo] ptr mem)
	// result: (CMPWconst (MOVWload {sym} [offOnly(vo)] ptr mem) [valOnly(vo)])
	for {
		vo := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(Op386CMPWconst)
		v.AuxInt = valOnly(vo)
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AuxInt = offOnly(vo)
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386splitload_Op386CMPWload_0(v *Value) bool {
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
		v.reset(Op386CMPW)
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteBlock386splitload(b *Block) bool {
	switch b.Kind {
	}
	return false
}
