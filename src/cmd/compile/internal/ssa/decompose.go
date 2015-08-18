// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// decompose converts phi ops on compound types into phi
// ops on simple types.
// (The remaining compound ops are decomposed with rewrite rules.)
func decompose(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			switch {
			case v.Type.IsString():
				decomposeStringPhi(v)
			case v.Type.IsSlice():
				decomposeSlicePhi(v)
			case v.Type.IsInterface():
				decomposeInterfacePhi(v)
				//case v.Type.IsStruct():
				//	decomposeStructPhi(v)
			case v.Type.Size() > f.Config.IntSize:
				f.Unimplementedf("undecomposed type %s", v.Type)
			}
		}
	}
	// TODO: decompose complex?
	// TODO: decompose 64-bit ops on 32-bit archs?
}

func decomposeStringPhi(v *Value) {
	fe := v.Block.Func.Config.fe
	ptrType := fe.TypeBytePtr()
	lenType := fe.TypeUintptr()

	ptr := v.Block.NewValue0(v.Line, OpPhi, ptrType)
	len := v.Block.NewValue0(v.Line, OpPhi, lenType)
	for _, a := range v.Args {
		ptr.AddArg(a.Block.NewValue1(v.Line, OpStringPtr, ptrType, a))
		len.AddArg(a.Block.NewValue1(v.Line, OpStringLen, lenType, a))
	}
	v.Op = OpStringMake
	v.AuxInt = 0
	v.Aux = nil
	v.resetArgs()
	v.AddArg(ptr)
	v.AddArg(len)
}

func decomposeSlicePhi(v *Value) {
	fe := v.Block.Func.Config.fe
	ptrType := fe.TypeBytePtr()
	lenType := fe.TypeUintptr()

	ptr := v.Block.NewValue0(v.Line, OpPhi, ptrType)
	len := v.Block.NewValue0(v.Line, OpPhi, lenType)
	cap := v.Block.NewValue0(v.Line, OpPhi, lenType)
	for _, a := range v.Args {
		ptr.AddArg(a.Block.NewValue1(v.Line, OpSlicePtr, ptrType, a))
		len.AddArg(a.Block.NewValue1(v.Line, OpSliceLen, lenType, a))
		cap.AddArg(a.Block.NewValue1(v.Line, OpSliceCap, lenType, a))
	}
	v.Op = OpSliceMake
	v.AuxInt = 0
	v.Aux = nil
	v.resetArgs()
	v.AddArg(ptr)
	v.AddArg(len)
	v.AddArg(cap)
}

func decomposeInterfacePhi(v *Value) {
	ptrType := v.Block.Func.Config.fe.TypeBytePtr()

	itab := v.Block.NewValue0(v.Line, OpPhi, ptrType)
	data := v.Block.NewValue0(v.Line, OpPhi, ptrType)
	for _, a := range v.Args {
		itab.AddArg(a.Block.NewValue1(v.Line, OpITab, ptrType, a))
		data.AddArg(a.Block.NewValue1(v.Line, OpIData, ptrType, a))
	}
	v.Op = OpIMake
	v.AuxInt = 0
	v.Aux = nil
	v.resetArgs()
	v.AddArg(itab)
	v.AddArg(data)
}
func decomposeStructPhi(v *Value) {
	// TODO
}
