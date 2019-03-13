// Code generated from gen/AMD64splitload.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "fmt"
import "math"
import "cmd/internal/obj"
import "cmd/internal/objabi"
import "cmd/compile/internal/types"

var _ = fmt.Println   // in case not otherwise used
var _ = math.MinInt8  // in case not otherwise used
var _ = obj.ANOP      // in case not otherwise used
var _ = objabi.GOROOT // in case not otherwise used
var _ = types.TypeMem // in case not otherwise used

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
	// cond:
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
	// cond:
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
	// cond:
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
	// cond:
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
	// cond:
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
	// cond:
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
	// cond:
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
	// cond:
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
	config := b.Func.Config
	typ := &config.Types
	_ = typ
	v := b.Control
	_ = v
	switch b.Kind {
	}
	return false
}
