// Code generated from _gen/dec.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValuedec(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpArrayMake1:
		return rewriteValuedec_OpArrayMake1(v)
	case ssaop.OpArraySelect:
		return rewriteValuedec_OpArraySelect(v)
	case ssaop.OpComplexImag:
		return rewriteValuedec_OpComplexImag(v)
	case ssaop.OpComplexReal:
		return rewriteValuedec_OpComplexReal(v)
	case ssaop.OpIData:
		return rewriteValuedec_OpIData(v)
	case ssaop.OpIMake:
		return rewriteValuedec_OpIMake(v)
	case ssaop.OpITab:
		return rewriteValuedec_OpITab(v)
	case ssaop.OpLoad:
		return rewriteValuedec_OpLoad(v)
	case ssaop.OpSliceCap:
		return rewriteValuedec_OpSliceCap(v)
	case ssaop.OpSliceLen:
		return rewriteValuedec_OpSliceLen(v)
	case ssaop.OpSlicePtr:
		return rewriteValuedec_OpSlicePtr(v)
	case ssaop.OpSlicePtrUnchecked:
		return rewriteValuedec_OpSlicePtrUnchecked(v)
	case ssaop.OpStore:
		return rewriteValuedec_OpStore(v)
	case ssaop.OpStringLen:
		return rewriteValuedec_OpStringLen(v)
	case ssaop.OpStringPtr:
		return rewriteValuedec_OpStringPtr(v)
	case ssaop.OpStructMake:
		return rewriteValuedec_OpStructMake(v)
	case ssaop.OpStructSelect:
		return rewriteValuedec_OpStructSelect(v)
	}
	return false
}
func rewriteValuedec_OpArrayMake1(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (ArrayMake1 x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValuedec_OpArraySelect(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ArraySelect [0] x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ArraySelect (ArrayMake1 x))
	// result: x
	for {
		if v_0.Op != ssaop.OpArrayMake1 {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (ArraySelect [0] (IData x))
	// result: (IData x)
	for {
		if AuxIntToInt64(v.AuxInt) != 0 || v_0.Op != ssaop.OpIData {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpIData)
		v.AddArg(x)
		return true
	}
	// match: (ArraySelect [i] x:(Load <t> ptr mem))
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.Elem().Size()*i] ptr) mem)
	for {
		i := AuxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, v.Type)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = Int64ToAuxInt(t.Elem().Size() * i)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpComplexImag(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ComplexImag (ComplexMake _ imag ))
	// result: imag
	for {
		if v_0.Op != ssaop.OpComplexMake {
			break
		}
		imag := v_0.Args[1]
		v.CopyOf(imag)
		return true
	}
	// match: (ComplexImag x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 8
	// result: @x.Block (Load <typ.Float32> (OffPtr <typ.Float32Ptr> [4] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float32)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float32Ptr)
		v1.AuxInt = Int64ToAuxInt(4)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (ComplexImag x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 16
	// result: @x.Block (Load <typ.Float64> (OffPtr <typ.Float64Ptr> [8] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float64)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float64Ptr)
		v1.AuxInt = Int64ToAuxInt(8)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpComplexReal(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ComplexReal (ComplexMake real _ ))
	// result: real
	for {
		if v_0.Op != ssaop.OpComplexMake {
			break
		}
		real := v_0.Args[0]
		v.CopyOf(real)
		return true
	}
	// match: (ComplexReal x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 8
	// result: @x.Block (Load <typ.Float32> ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 8) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float32)
		v.CopyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (ComplexReal x:(Load <t> ptr mem))
	// cond: t.IsComplex() && t.Size() == 16
	// result: @x.Block (Load <typ.Float64> ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsComplex() && t.Size() == 16) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float64)
		v.CopyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpIData(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (IData (IMake _ data))
	// cond: data.Op != ssaop.OpStructMake && data.Op != ssaop.OpArrayMake1
	// result: data
	for {
		if v_0.Op != ssaop.OpIMake {
			break
		}
		data := v_0.Args[1]
		if !(data.Op != ssaop.OpStructMake && data.Op != ssaop.OpArrayMake1) {
			break
		}
		v.CopyOf(data)
		return true
	}
	// match: (IData x:(Load <t> ptr mem))
	// cond: t.IsInterface()
	// result: @x.Block (Load <typ.BytePtr> (OffPtr <typ.BytePtrPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsInterface()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.BytePtr)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.BytePtrPtr)
		v1.AuxInt = Int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpIMake(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IMake _typ (StructMake ___))
	// result: ImakeOfStructMake(v)
	for {
		if v_1.Op != ssaop.OpStructMake {
			break
		}
		v.CopyOf(ImakeOfStructMake(v))
		return true
	}
	// match: (IMake _typ (ArrayMake1 val))
	// result: (IMake _typ val)
	for {
		_typ := v_0
		if v_1.Op != ssaop.OpArrayMake1 {
			break
		}
		val := v_1.Args[0]
		v.Reset(ssaop.OpIMake)
		v.AddArg2(_typ, val)
		return true
	}
	return false
}
func rewriteValuedec_OpITab(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ITab (IMake itab _))
	// result: itab
	for {
		if v_0.Op != ssaop.OpIMake {
			break
		}
		itab := v_0.Args[0]
		v.CopyOf(itab)
		return true
	}
	// match: (ITab x:(Load <t> ptr mem))
	// cond: t.IsInterface()
	// result: @x.Block (Load <typ.Uintptr> ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsInterface()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Uintptr)
		v.CopyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpLoad(v *ssacore.Value) bool {
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
		v.Reset(ssaop.OpComplexMake)
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float32)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float32)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float32Ptr)
		v2.AuxInt = Int64ToAuxInt(4)
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
		v.Reset(ssaop.OpComplexMake)
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float64)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Float64)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float64Ptr)
		v2.AuxInt = Int64ToAuxInt(8)
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
		v.Reset(ssaop.OpStringMake)
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.BytePtr)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v2.AuxInt = Int64ToAuxInt(config.PtrSize)
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
		v.Reset(ssaop.OpSliceMake)
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, t.Elem().PtrTo())
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v2.AuxInt = Int64ToAuxInt(config.PtrSize)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v4 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v4.AuxInt = Int64ToAuxInt(2 * config.PtrSize)
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
		v.Reset(ssaop.OpIMake)
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Uintptr)
		v0.AddArg2(ptr, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.BytePtr)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.BytePtrPtr)
		v2.AuxInt = Int64ToAuxInt(config.PtrSize)
		v2.AddArg(ptr)
		v1.AddArg2(v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceCap(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (SliceCap (SliceMake _ _ cap))
	// result: cap
	for {
		if v_0.Op != ssaop.OpSliceMake {
			break
		}
		cap := v_0.Args[2]
		v.CopyOf(cap)
		return true
	}
	// match: (SliceCap x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [2*config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v1.AuxInt = Int64ToAuxInt(2 * config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSliceLen(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (SliceLen (SliceMake _ len _))
	// result: len
	for {
		if v_0.Op != ssaop.OpSliceMake {
			break
		}
		len := v_0.Args[1]
		v.CopyOf(len)
		return true
	}
	// match: (SliceLen x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v1.AuxInt = Int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSlicePtr(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SlicePtr (SliceMake ptr _ _ ))
	// result: ptr
	for {
		if v_0.Op != ssaop.OpSliceMake {
			break
		}
		ptr := v_0.Args[0]
		v.CopyOf(ptr)
		return true
	}
	// match: (SlicePtr x:(Load <t> ptr mem))
	// cond: t.IsSlice()
	// result: @x.Block (Load <t.Elem().PtrTo()> ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsSlice()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, t.Elem().PtrTo())
		v.CopyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpSlicePtrUnchecked(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (SlicePtrUnchecked (SliceMake ptr _ _ ))
	// result: ptr
	for {
		if v_0.Op != ssaop.OpSliceMake {
			break
		}
		ptr := v_0.Args[0]
		v.CopyOf(ptr)
		return true
	}
	return false
}
func rewriteValuedec_OpStore(v *ssacore.Value) bool {
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
		t := AuxToType(v.Aux)
		mem := v_2
		if !(t.Size() == 0) {
			break
		}
		v.CopyOf(mem)
		return true
	}
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.Size() == 8
	// result: (Store {typ.Float32} (OffPtr <typ.Float32Ptr> [4] dst) imag (Store {typ.Float32} dst real mem))
	for {
		t := AuxToType(v.Aux)
		dst := v_0
		if v_1.Op != ssaop.OpComplexMake {
			break
		}
		imag := v_1.Args[1]
		real := v_1.Args[0]
		mem := v_2
		if !(t.Size() == 8) {
			break
		}
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(typ.Float32)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float32Ptr)
		v0.AuxInt = Int64ToAuxInt(4)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v1.Aux = TypeToAux(typ.Float32)
		v1.AddArg3(dst, real, mem)
		v.AddArg3(v0, imag, v1)
		return true
	}
	// match: (Store {t} dst (ComplexMake real imag) mem)
	// cond: t.Size() == 16
	// result: (Store {typ.Float64} (OffPtr <typ.Float64Ptr> [8] dst) imag (Store {typ.Float64} dst real mem))
	for {
		t := AuxToType(v.Aux)
		dst := v_0
		if v_1.Op != ssaop.OpComplexMake {
			break
		}
		imag := v_1.Args[1]
		real := v_1.Args[0]
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(typ.Float64)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.Float64Ptr)
		v0.AuxInt = Int64ToAuxInt(8)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v1.Aux = TypeToAux(typ.Float64)
		v1.AddArg3(dst, real, mem)
		v.AddArg3(v0, imag, v1)
		return true
	}
	// match: (Store dst (StringMake ptr len) mem)
	// result: (Store {typ.Int} (OffPtr <typ.IntPtr> [config.PtrSize] dst) len (Store {typ.BytePtr} dst ptr mem))
	for {
		dst := v_0
		if v_1.Op != ssaop.OpStringMake {
			break
		}
		len := v_1.Args[1]
		ptr := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(typ.Int)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v0.AuxInt = Int64ToAuxInt(config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v1.Aux = TypeToAux(typ.BytePtr)
		v1.AddArg3(dst, ptr, mem)
		v.AddArg3(v0, len, v1)
		return true
	}
	// match: (Store {t} dst (SliceMake ptr len cap) mem)
	// result: (Store {typ.Int} (OffPtr <typ.IntPtr> [2*config.PtrSize] dst) cap (Store {typ.Int} (OffPtr <typ.IntPtr> [config.PtrSize] dst) len (Store {t.Elem().PtrTo()} dst ptr mem)))
	for {
		t := AuxToType(v.Aux)
		dst := v_0
		if v_1.Op != ssaop.OpSliceMake {
			break
		}
		cap := v_1.Args[2]
		ptr := v_1.Args[0]
		len := v_1.Args[1]
		mem := v_2
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(typ.Int)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v0.AuxInt = Int64ToAuxInt(2 * config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v1.Aux = TypeToAux(typ.Int)
		v2 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v2.AuxInt = Int64ToAuxInt(config.PtrSize)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v3.Aux = TypeToAux(t.Elem().PtrTo())
		v3.AddArg3(dst, ptr, mem)
		v1.AddArg3(v2, len, v3)
		v.AddArg3(v0, cap, v1)
		return true
	}
	// match: (Store dst (IMake itab data) mem)
	// result: (Store {typ.BytePtr} (OffPtr <typ.BytePtrPtr> [config.PtrSize] dst) data (Store {typ.Uintptr} dst itab mem))
	for {
		dst := v_0
		if v_1.Op != ssaop.OpIMake {
			break
		}
		data := v_1.Args[1]
		itab := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(typ.BytePtr)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.BytePtrPtr)
		v0.AuxInt = Int64ToAuxInt(config.PtrSize)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.OpStore, types.TypeMem)
		v1.Aux = TypeToAux(typ.Uintptr)
		v1.AddArg3(dst, itab, mem)
		v.AddArg3(v0, data, v1)
		return true
	}
	// match: (Store _ (StructMake ___) _)
	// result: RewriteStructStore(v)
	for {
		if v_1.Op != ssaop.OpStructMake {
			break
		}
		v.CopyOf(RewriteStructStore(v))
		return true
	}
	// match: (Store dst (ArrayMake1 e) mem)
	// result: (Store {e.Type} dst e mem)
	for {
		dst := v_0
		if v_1.Op != ssaop.OpArrayMake1 {
			break
		}
		e := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpStore)
		v.Aux = TypeToAux(e.Type)
		v.AddArg3(dst, e, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStringLen(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (StringLen (StringMake _ len))
	// result: len
	for {
		if v_0.Op != ssaop.OpStringMake {
			break
		}
		len := v_0.Args[1]
		v.CopyOf(len)
		return true
	}
	// match: (StringLen x:(Load <t> ptr mem))
	// cond: t.IsString()
	// result: @x.Block (Load <typ.Int> (OffPtr <typ.IntPtr> [config.PtrSize] ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsString()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.Int)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, typ.IntPtr)
		v1.AuxInt = Int64ToAuxInt(config.PtrSize)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStringPtr(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (StringPtr (StringMake ptr _))
	// result: ptr
	for {
		if v_0.Op != ssaop.OpStringMake {
			break
		}
		ptr := v_0.Args[0]
		v.CopyOf(ptr)
		return true
	}
	// match: (StringPtr x:(Load <t> ptr mem))
	// cond: t.IsString()
	// result: @x.Block (Load <typ.BytePtr> ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(t.IsString()) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, typ.BytePtr)
		v.CopyOf(v0)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuedec_OpStructMake(v *ssacore.Value) bool {
	// match: (StructMake x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		if len(v.Args) != 1 {
			break
		}
		x := v.Args[0]
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValuedec_OpStructSelect(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (StructSelect (IData x))
	// cond: v.Type.Size() > 0
	// result: (IData x)
	for {
		if v_0.Op != ssaop.OpIData {
			break
		}
		x := v_0.Args[0]
		if !(v.Type.Size() > 0) {
			break
		}
		v.Reset(ssaop.OpIData)
		v.AddArg(x)
		return true
	}
	// match: (StructSelect (IData x))
	// cond: v.Type.Size() == 0
	// result: (Empty)
	for {
		if v_0.Op != ssaop.OpIData {
			break
		}
		if !(v.Type.Size() == 0) {
			break
		}
		v.Reset(ssaop.OpEmpty)
		return true
	}
	// match: (StructSelect [i] x:(StructMake ___))
	// result: x.Args[i]
	for {
		i := AuxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpStructMake {
			break
		}
		v.CopyOf(x.Args[i])
		return true
	}
	// match: (StructSelect x)
	// cond: x.Type.IsPtrShaped()
	// result: x
	for {
		x := v_0
		if !(x.Type.IsPtrShaped()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (StructSelect [i] x:(Load <t> ptr mem))
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.FieldOff(int(i))] ptr) mem)
	for {
		i := AuxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpLoad, v.Type)
		v.CopyOf(v0)
		v1 := b.NewValue0(v.Pos, ssaop.OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = Int64ToAuxInt(t.FieldOff(int(i)))
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	return false
}
func rewriteBlockdec(b *ssacore.Block) bool {
	return false
}
