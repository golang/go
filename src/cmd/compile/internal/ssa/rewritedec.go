// Code generated from gen/dec.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/internal/obj"
import "cmd/internal/objabi"
import "cmd/compile/internal/types"

var _ = math.MinInt8  // in case not otherwise used
var _ = obj.ANOP      // in case not otherwise used
var _ = objabi.GOROOT // in case not otherwise used
var _ = types.TypeMem // in case not otherwise used

func rewriteValuedec(v *Value) bool {
	switch v.Op {
	case OpComplexImag:
		return rewriteValuedec_OpComplexImag_0(v)
	case OpComplexReal:
		return rewriteValuedec_OpComplexReal_0(v)
	case OpIData:
		return rewriteValuedec_OpIData_0(v)
	case OpITab:
		return rewriteValuedec_OpITab_0(v)
	case OpLoad:
		return rewriteValuedec_OpLoad_0(v)
	case OpSliceCap:
		return rewriteValuedec_OpSliceCap_0(v)
	case OpSliceLen:
		return rewriteValuedec_OpSliceLen_0(v)
	case OpSlicePtr:
		return rewriteValuedec_OpSlicePtr_0(v)
	case OpStore:
		return rewriteValuedec_OpStore_0(v)
	case OpStringLen:
		return rewriteValuedec_OpStringLen_0(v)
	case OpStringPtr:
		return rewriteValuedec_OpStringPtr_0(v)
	}
	return false
}
func rewriteValuedec_OpComplexImag_0(v *Value) bool {
	// match: (ComplexImag (ComplexMake _ imag))
	// cond:
	// result: imag
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpComplexMake {
			break
		}
		_ = v_0.Args[1]
		imag := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = imag.Type
		v.AddArg(imag)
		return true
	}
	return false
}
func rewriteValuedec_OpComplexReal_0(v *Value) bool {
	// match: (ComplexReal (ComplexMake real _))
	// cond:
	// result: real
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpComplexMake {
			break
		}
		_ = v_0.Args[1]
		real := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = real.Type
		v.AddArg(real)
		return true
	}
	return false
}
func rewriteValuedec_OpIData_0(v *Value) bool {
	// match: (IData (IMake _ data))
	// cond:
	// result: data
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpIMake {
			break
		}
		_ = v_0.Args[1]
		data := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = data.Type
		v.AddArg(data)
		return true
	}
	return false
}
func rewriteValuedec_OpITab_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ITab (IMake itab _))
	// cond:
	// result: itab
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpIMake {
			break
		}
		_ = v_0.Args[1]
		itab := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = itab.Type
		v.AddArg(itab)
		return true
	}
	return false
}
func rewriteValuedec_OpLoad_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Load <t> ptr mem)
	// cond: t.IsComplex() && t.Size() == 8
	// result: (ComplexMake     (Load <typ.Float32> ptr mem)     (Load <typ.Float32>       (OffPtr <typ.Float32Ptr> [4] ptr)       mem)     )
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.Float32Ptr)
		v2.AuxInt = 4
		v2.AddArg(ptr)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsComplex() && t.Size() == 16
	// result: (ComplexMake     (Load <typ.Float64> ptr mem)     (Load <typ.Float64>       (OffPtr <typ.Float64Ptr> [8] ptr)       mem)     )
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.Float64Ptr)
		v2.AuxInt = 8
		v2.AddArg(ptr)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsString()
	// result: (StringMake     (Load <typ.BytePtr> ptr mem)     (Load <typ.Int>       (OffPtr <typ.IntPtr> [config.PtrSize] ptr)       mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsString()) {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = config.PtrSize
		v2.AddArg(ptr)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsSlice()
	// result: (SliceMake     (Load <t.ElemType().PtrTo()> ptr mem)     (Load <typ.Int>       (OffPtr <typ.IntPtr> [config.PtrSize] ptr)       mem)     (Load <typ.Int>       (OffPtr <typ.IntPtr> [2*config.PtrSize] ptr)       mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsSlice()) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpLoad, t.ElemType().PtrTo())
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = config.PtrSize
		v2.AddArg(ptr)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v4 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v4.AuxInt = 2 * config.PtrSize
		v4.AddArg(ptr)
		v3.AddArg(v4)
		v3.AddArg(mem)
		v.AddArg(v3)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsInterface()
	// result: (IMake     (Load <typ.BytePtr> ptr mem)     (Load <typ.BytePtr>       (OffPtr <typ.BytePtrPtr> [config.PtrSize] ptr)       mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsInterface()) {
			break
		}
		v.reset(OpIMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.BytePtrPtr)
		v2.AuxInt = config.PtrSize
		v2.AddArg(ptr)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceCap_0(v *Value) bool {
	// match: (SliceCap (SliceMake _ _ cap))
	// cond:
	// result: cap
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		cap := v_0.Args[2]
		v.reset(OpCopy)
		v.Type = cap.Type
		v.AddArg(cap)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceLen_0(v *Value) bool {
	// match: (SliceLen (SliceMake _ len _))
	// cond:
	// result: len
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		len := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = len.Type
		v.AddArg(len)
		return true
	}
	return false
}
func rewriteValuedec_OpSlicePtr_0(v *Value) bool {
	// match: (SlicePtr (SliceMake ptr _ _))
	// cond:
	// result: ptr
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		ptr := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = ptr.Type
		v.AddArg(ptr)
		return true
	}
	return false
}
func rewriteValuedec_OpStore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.(*types.Type).Size() == 8
	// result: (Store {typ.Float32}     (OffPtr <typ.Float32Ptr> [4] dst)     imag     (Store {typ.Float32} dst real mem))
	for {
		t := v.Aux
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpComplexMake {
			break
		}
		_ = v_1.Args[1]
		real := v_1.Args[0]
		imag := v_1.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 8) {
			break
		}
		v.reset(OpStore)
		v.Aux = typ.Float32
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.Float32Ptr)
		v0.AuxInt = 4
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(imag)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typ.Float32
		v1.AddArg(dst)
		v1.AddArg(real)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.(*types.Type).Size() == 16
	// result: (Store {typ.Float64}     (OffPtr <typ.Float64Ptr> [8] dst)     imag     (Store {typ.Float64} dst real mem))
	for {
		t := v.Aux
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpComplexMake {
			break
		}
		_ = v_1.Args[1]
		real := v_1.Args[0]
		imag := v_1.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 16) {
			break
		}
		v.reset(OpStore)
		v.Aux = typ.Float64
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.Float64Ptr)
		v0.AuxInt = 8
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(imag)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typ.Float64
		v1.AddArg(dst)
		v1.AddArg(real)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Store dst (StringMake ptr len) mem)
	// cond:
	// result: (Store {typ.Int}     (OffPtr <typ.IntPtr> [config.PtrSize] dst)     len     (Store {typ.BytePtr} dst ptr mem))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStringMake {
			break
		}
		_ = v_1.Args[1]
		ptr := v_1.Args[0]
		len := v_1.Args[1]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = typ.Int
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v0.AuxInt = config.PtrSize
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(len)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typ.BytePtr
		v1.AddArg(dst)
		v1.AddArg(ptr)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Store dst (SliceMake ptr len cap) mem)
	// cond:
	// result: (Store {typ.Int}     (OffPtr <typ.IntPtr> [2*config.PtrSize] dst)     cap     (Store {typ.Int}       (OffPtr <typ.IntPtr> [config.PtrSize] dst)       len       (Store {typ.BytePtr} dst ptr mem)))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSliceMake {
			break
		}
		_ = v_1.Args[2]
		ptr := v_1.Args[0]
		len := v_1.Args[1]
		cap := v_1.Args[2]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = typ.Int
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v0.AuxInt = 2 * config.PtrSize
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(cap)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typ.Int
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = config.PtrSize
		v2.AddArg(dst)
		v1.AddArg(v2)
		v1.AddArg(len)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typ.BytePtr
		v3.AddArg(dst)
		v3.AddArg(ptr)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Store dst (IMake itab data) mem)
	// cond:
	// result: (Store {typ.BytePtr}     (OffPtr <typ.BytePtrPtr> [config.PtrSize] dst)     data     (Store {typ.Uintptr} dst itab mem))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpIMake {
			break
		}
		_ = v_1.Args[1]
		itab := v_1.Args[0]
		data := v_1.Args[1]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = typ.BytePtr
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.BytePtrPtr)
		v0.AuxInt = config.PtrSize
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(data)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typ.Uintptr
		v1.AddArg(dst)
		v1.AddArg(itab)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValuedec_OpStringLen_0(v *Value) bool {
	// match: (StringLen (StringMake _ len))
	// cond:
	// result: len
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpStringMake {
			break
		}
		_ = v_0.Args[1]
		len := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = len.Type
		v.AddArg(len)
		return true
	}
	return false
}
func rewriteValuedec_OpStringPtr_0(v *Value) bool {
	// match: (StringPtr (StringMake ptr _))
	// cond:
	// result: ptr
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpStringMake {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = ptr.Type
		v.AddArg(ptr)
		return true
	}
	return false
}
func rewriteBlockdec(b *Block) bool {
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &config.Types
	_ = typ
	switch b.Kind {
	}
	return false
}
