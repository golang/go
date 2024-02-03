// Code generated from _gen/dec.rules using 'go generate'; DO NOT EDIT.

package ssa

import "cmd/compile/internal/types"

func rewriteValuedec(v *Value) bool {
	switch v.Op {
	case OpArrayMake1:
		return rewriteValuedec_OpArrayMake1(v)
	case OpArraySelect:
		return rewriteValuedec_OpArraySelect(v)
	case OpComplexImag:
		return rewriteValuedec_OpComplexImag(v)
	case OpComplexReal:
		return rewriteValuedec_OpComplexReal(v)
	case OpIData:
		return rewriteValuedec_OpIData(v)
	case OpIMake:
		return rewriteValuedec_OpIMake(v)
	case OpITab:
		return rewriteValuedec_OpITab(v)
	case OpLoad:
		return rewriteValuedec_OpLoad(v)
	case OpSliceCap:
		return rewriteValuedec_OpSliceCap(v)
	case OpSliceLen:
		return rewriteValuedec_OpSliceLen(v)
	case OpSlicePtr:
		return rewriteValuedec_OpSlicePtr(v)
	case OpSlicePtrUnchecked:
		return rewriteValuedec_OpSlicePtrUnchecked(v)
	case OpStore:
		return rewriteValuedec_OpStore(v)
	case OpStringLen:
		return rewriteValuedec_OpStringLen(v)
	case OpStringPtr:
		return rewriteValuedec_OpStringPtr(v)
	case OpStructMake1:
		return rewriteValuedec_OpStructMake1(v)
	case OpStructSelect:
		return rewriteValuedec_OpStructSelect(v)
	}
	return false
}
func rewriteValuedec_OpArrayMake1(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ArrayMake1 x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuedec_OpArraySelect(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ArraySelect [0] x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ArraySelect (ArrayMake1 x))
	// result: x
	for {
		if v_0.Op != OpArrayMake1 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (ArraySelect [0] (IData x))
	// result: (IData x)
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpIData {
			break
		}
		x := v_0.Args[0]
		v.reset(OpIData)
		v.AddArg(x)
		return true
	}
	// match: (ArraySelect [i] x:(Load <t> ptr mem))
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.Elem().Size()*i] ptr) mem)
	for {
		i := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, v.Type)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = int64ToAuxInt(t.Elem().Size() * i)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpComplexImag(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ComplexImag (ComplexMake _ imag ))
	// result: imag
	for {
		if v_0.Op != OpComplexMake {
			break
		}
		imag := v_0.Args[1]
		v.copyOf(imag)
		return true
	}
	// match: (ComplexImag x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 8
	// result: @x.Block (Load <typ.Float32> (OffPtr <typ.Float32Ptr> [4] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.Float32Ptr)
		v1.AuxInt = int64ToAuxInt(4)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (ComplexImag x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 16
	// result: @x.Block (Load <typ.Float64> (OffPtr <typ.Float64Ptr> [8] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.Float64Ptr)
		v1.AuxInt = int64ToAuxInt(8)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpComplexReal(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ComplexReal (ComplexMake real _ ))
	// result: real
	for {
		if v_0.Op != OpComplexMake {
			break
		}
		real := v_0.Args[0]
		v.copyOf(real)
		return true
	}
	// match: (ComplexReal x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 8
	// result: @x.Block (Load <typ.Float32> ptr mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v.copyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (ComplexReal x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 16
	// result: @x.Block (Load <typ.Float64> ptr mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v.copyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpIData(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (IData (IMake _ data))
	// result: data
	for {
		if v_0.Op != OpIMake {
			break
		}
		data := v_0.Args[1]
		v.copyOf(data)
		return true
	}
	// match: (IData x:(Load <t> ptr mem))
	// cond: t.IsInterface()
	// result: @x.Block (Load <typ.BytePtr> (OffPtr <typ.BytePtrPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsInterface()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.BytePtrPtr)
		v1.AuxInt = int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpIMake(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IMake _typ (StructMake1 val))
	// result: (IMake _typ val)
	for {
		_typ := v_0
		if v_1.Op != OpStructMake1 {
			break
		}
		val := v_1.Args[0]
		v.reset(OpIMake)
		v.AddArg2(_typ, val)
		return true
	}
	return false
}
func rewriteValuedec_OpITab(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ITab (IMake itab _))
	// result: itab
	for {
		if v_0.Op != OpIMake {
			break
		}
		itab := v_0.Args[0]
		v.copyOf(itab)
		return true
	}
	// match: (ITab x:(Load <t> ptr mem))
	// cond: t.IsInterface()
	// result: @x.Block (Load <typ.Uintptr> ptr mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsInterface()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Uintptr)
		v.copyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Load <t> ptr mem)
	// cond: t.IsComplex() && t.Size() == 8
	// result: (ComplexMake (Load <typ.Float32> ptr mem) (Load <typ.Float32> (OffPtr <typ.Float32Ptr> [4] ptr) mem) )
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Float32)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.Float32Ptr)
		v2.AuxInt = int64ToAuxInt(4)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsComplex() && t.Size() == 16
	// result: (ComplexMake (Load <typ.Float64> ptr mem) (Load <typ.Float64> (OffPtr <typ.Float64Ptr> [8] ptr) mem) )
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Float64)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.Float64Ptr)
		v2.AuxInt = int64ToAuxInt(8)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsString()
	// result: (StringMake (Load <typ.BytePtr> ptr mem) (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsString()) {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = int64ToAuxInt(config.PtrSize)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsSlice()
	// result: (SliceMake (Load <t.Elem().PtrTo()> ptr mem) (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem) (Load <typ.Int> (OffPtr <typ.IntPtr> [2*config.PtrSize] ptr) mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsSlice()) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpLoad, t.Elem().PtrTo())
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = int64ToAuxInt(config.PtrSize)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v3 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v4 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v4.AuxInt = int64ToAuxInt(2 * config.PtrSize)
		v4.AddArg(ptr)
		v3.AddArg2(v4, mem)
		v.AddArg3(v0, v1, v3)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsInterface()
	// result: (IMake (Load <typ.Uintptr> ptr mem) (Load <typ.BytePtr> (OffPtr <typ.BytePtrPtr> [config.PtrSize] ptr) mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsInterface()) {
			break
		}
		v.reset(OpIMake)
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Uintptr)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.BytePtrPtr)
		v2.AuxInt = int64ToAuxInt(config.PtrSize)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceCap(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (SliceCap (SliceMake _ _ cap))
	// result: cap
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		cap := v_0.Args[2]
		v.copyOf(cap)
		return true
	}
	// match: (SliceCap x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [2*config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v1.AuxInt = int64ToAuxInt(2 * config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceLen(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (SliceLen (SliceMake _ len _))
	// result: len
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		len := v_0.Args[1]
		v.copyOf(len)
		return true
	}
	// match: (SliceLen x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v1.AuxInt = int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSlicePtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SlicePtr (SliceMake ptr _ _ ))
	// result: ptr
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		ptr := v_0.Args[0]
		v.copyOf(ptr)
		return true
	}
	// match: (SlicePtr x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <t.Elem().PtrTo()> ptr mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, t.Elem().PtrTo())
		v.copyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSlicePtrUnchecked(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SlicePtrUnchecked (SliceMake ptr _ _ ))
	// result: ptr
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		ptr := v_0.Args[0]
		v.copyOf(ptr)
		return true
	}
	return false
}
func rewriteValuedec_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Store {t} _ _ mem)
	// cond: t.Size() == 0
	// result: mem
	for {
		t := auxToType(v.Aux)
		mem := v_2
		if !(t.Size() == 0) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.Size() == 8
	// result: (Store {typ.Float32} (OffPtr <typ.Float32Ptr> [4] dst) imag (Store {typ.Float32} dst real mem))
	for {
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpComplexMake {
			break
		}
		imag := v_1.Args[1]
		real := v_1.Args[0]
		mem := v_2
		if !(t.Size() == 8) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(typ.Float32)
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.Float32Ptr)
		v0.AuxInt = int64ToAuxInt(4)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(typ.Float32)
		v1.AddArg3(dst, real, mem)
		v.AddArg3(v0, imag, v1)
		return true
	}
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.Size() == 16
	// result: (Store {typ.Float64} (OffPtr <typ.Float64Ptr> [8] dst) imag (Store {typ.Float64} dst real mem))
	for {
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpComplexMake {
			break
		}
		imag := v_1.Args[1]
		real := v_1.Args[0]
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(typ.Float64)
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.Float64Ptr)
		v0.AuxInt = int64ToAuxInt(8)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(typ.Float64)
		v1.AddArg3(dst, real, mem)
		v.AddArg3(v0, imag, v1)
		return true
	}
	// match: (Store dst (StringMake ptr len) mem)
	// result: (Store {typ.Int} (OffPtr <typ.IntPtr> [config.PtrSize] dst) len (Store {typ.BytePtr} dst ptr mem))
	for {
		dst := v_0
		if v_1.Op != OpStringMake {
			break
		}
		len := v_1.Args[1]
		ptr := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(typ.Int)
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v0.AuxInt = int64ToAuxInt(config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(typ.BytePtr)
		v1.AddArg3(dst, ptr, mem)
		v.AddArg3(v0, len, v1)
		return true
	}
	// match: (Store {t} dst (SliceMake ptr len cap) mem)
	// result: (Store {typ.Int} (OffPtr <typ.IntPtr> [2*config.PtrSize] dst) cap (Store {typ.Int} (OffPtr <typ.IntPtr> [config.PtrSize] dst) len (Store {t.Elem().PtrTo()} dst ptr mem)))
	for {
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpSliceMake {
			break
		}
		cap := v_1.Args[2]
		ptr := v_1.Args[0]
		len := v_1.Args[1]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(typ.Int)
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v0.AuxInt = int64ToAuxInt(2 * config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(typ.Int)
		v2 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v2.AuxInt = int64ToAuxInt(config.PtrSize)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t.Elem().PtrTo())
		v3.AddArg3(dst, ptr, mem)
		v1.AddArg3(v2, len, v3)
		v.AddArg3(v0, cap, v1)
		return true
	}
	// match: (Store dst (IMake itab data) mem)
	// result: (Store {typ.BytePtr} (OffPtr <typ.BytePtrPtr> [config.PtrSize] dst) data (Store {typ.Uintptr} dst itab mem))
	for {
		dst := v_0
		if v_1.Op != OpIMake {
			break
		}
		data := v_1.Args[1]
		itab := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(typ.BytePtr)
		v0 := b.NewValue0(v.Pos, OpOffPtr, typ.BytePtrPtr)
		v0.AuxInt = int64ToAuxInt(config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(typ.Uintptr)
		v1.AddArg3(dst, itab, mem)
		v.AddArg3(v0, data, v1)
		return true
	}
	// match: (Store dst (StructMake1 <t> f0) mem)
	// result: (Store {t.FieldType(0)} (OffPtr <t.FieldType(0).PtrTo()> [0] dst) f0 mem)
	for {
		dst := v_0
		if v_1.Op != OpStructMake1 {
			break
		}
		t := v_1.Type
		f0 := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(t.FieldType(0))
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v0.AuxInt = int64ToAuxInt(0)
		v0.AddArg(dst)
		v.AddArg3(v0, f0, mem)
		return true
	}
	// match: (Store dst (StructMake2 <t> f0 f1) mem)
	// result: (Store {t.FieldType(1)} (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst) f1 (Store {t.FieldType(0)} (OffPtr <t.FieldType(0).PtrTo()> [0] dst) f0 mem))
	for {
		dst := v_0
		if v_1.Op != OpStructMake2 {
			break
		}
		t := v_1.Type
		f1 := v_1.Args[1]
		f0 := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(t.FieldType(1))
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v0.AuxInt = int64ToAuxInt(t.FieldOff(1))
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t.FieldType(0))
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v2.AuxInt = int64ToAuxInt(0)
		v2.AddArg(dst)
		v1.AddArg3(v2, f0, mem)
		v.AddArg3(v0, f1, v1)
		return true
	}
	// match: (Store dst (StructMake3 <t> f0 f1 f2) mem)
	// result: (Store {t.FieldType(2)} (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] dst) f2 (Store {t.FieldType(1)} (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst) f1 (Store {t.FieldType(0)} (OffPtr <t.FieldType(0).PtrTo()> [0] dst) f0 mem)))
	for {
		dst := v_0
		if v_1.Op != OpStructMake3 {
			break
		}
		t := v_1.Type
		f2 := v_1.Args[2]
		f0 := v_1.Args[0]
		f1 := v_1.Args[1]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(t.FieldType(2))
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v0.AuxInt = int64ToAuxInt(t.FieldOff(2))
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t.FieldType(1))
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v2.AuxInt = int64ToAuxInt(t.FieldOff(1))
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t.FieldType(0))
		v4 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v4.AuxInt = int64ToAuxInt(0)
		v4.AddArg(dst)
		v3.AddArg3(v4, f0, mem)
		v1.AddArg3(v2, f1, v3)
		v.AddArg3(v0, f2, v1)
		return true
	}
	// match: (Store dst (StructMake4 <t> f0 f1 f2 f3) mem)
	// result: (Store {t.FieldType(3)} (OffPtr <t.FieldType(3).PtrTo()> [t.FieldOff(3)] dst) f3 (Store {t.FieldType(2)} (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] dst) f2 (Store {t.FieldType(1)} (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst) f1 (Store {t.FieldType(0)} (OffPtr <t.FieldType(0).PtrTo()> [0] dst) f0 mem))))
	for {
		dst := v_0
		if v_1.Op != OpStructMake4 {
			break
		}
		t := v_1.Type
		f3 := v_1.Args[3]
		f0 := v_1.Args[0]
		f1 := v_1.Args[1]
		f2 := v_1.Args[2]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(t.FieldType(3))
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(3).PtrTo())
		v0.AuxInt = int64ToAuxInt(t.FieldOff(3))
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t.FieldType(2))
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v2.AuxInt = int64ToAuxInt(t.FieldOff(2))
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t.FieldType(1))
		v4 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v4.AuxInt = int64ToAuxInt(t.FieldOff(1))
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = typeToAux(t.FieldType(0))
		v6 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v6.AuxInt = int64ToAuxInt(0)
		v6.AddArg(dst)
		v5.AddArg3(v6, f0, mem)
		v3.AddArg3(v4, f1, v5)
		v1.AddArg3(v2, f2, v3)
		v.AddArg3(v0, f3, v1)
		return true
	}
	// match: (Store dst (ArrayMake1 e) mem)
	// result: (Store {e.Type} dst e mem)
	for {
		dst := v_0
		if v_1.Op != OpArrayMake1 {
			break
		}
		e := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(e.Type)
		v.AddArg3(dst, e, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStringLen(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (StringLen (StringMake _ len))
	// result: len
	for {
		if v_0.Op != OpStringMake {
			break
		}
		len := v_0.Args[1]
		v.copyOf(len)
		return true
	}
	// match: (StringLen x:(Load <t> ptr mem))
	// cond: t.IsString()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsString()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.Int)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, typ.IntPtr)
		v1.AuxInt = int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStringPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (StringPtr (StringMake ptr _))
	// result: ptr
	for {
		if v_0.Op != OpStringMake {
			break
		}
		ptr := v_0.Args[0]
		v.copyOf(ptr)
		return true
	}
	// match: (StringPtr x:(Load <t> ptr mem))
	// cond: t.IsString()
	// result: @x.Block (Load <typ.BytePtr> ptr mem)
	for {
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsString()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, typ.BytePtr)
		v.copyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStructMake1(v *Value) bool {
	v_0 := v.Args[0]
	// match: (StructMake1 x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuedec_OpStructSelect(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (StructSelect [0] (IData x))
	// result: (IData x)
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpIData {
			break
		}
		x := v_0.Args[0]
		v.reset(OpIData)
		v.AddArg(x)
		return true
	}
	// match: (StructSelect (StructMake1 x))
	// result: x
	for {
		if v_0.Op != OpStructMake1 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [0] (StructMake2 x _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpStructMake2 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [1] (StructMake2 _ x))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpStructMake2 {
			break
		}
		x := v_0.Args[1]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [0] (StructMake3 x _ _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpStructMake3 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [1] (StructMake3 _ x _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpStructMake3 {
			break
		}
		x := v_0.Args[1]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [2] (StructMake3 _ _ x))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 2 || v_0.Op != OpStructMake3 {
			break
		}
		x := v_0.Args[2]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [0] (StructMake4 x _ _ _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpStructMake4 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [1] (StructMake4 _ x _ _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpStructMake4 {
			break
		}
		x := v_0.Args[1]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [2] (StructMake4 _ _ x _))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 2 || v_0.Op != OpStructMake4 {
			break
		}
		x := v_0.Args[2]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [3] (StructMake4 _ _ _ x))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 3 || v_0.Op != OpStructMake4 {
			break
		}
		x := v_0.Args[3]
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [0] x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (StructSelect [i] x:(Load <t> ptr mem))
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.FieldOff(int(i))] ptr) mem)
	for {
		i := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, v.Type)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = int64ToAuxInt(t.FieldOff(int(i)))
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteBlockdec(b *Block) bool {
	return false
}
