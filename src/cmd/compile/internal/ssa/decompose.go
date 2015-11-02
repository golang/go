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
			case v.Type.IsComplex():
				decomposeComplexPhi(v)
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
	// TODO: decompose 64-bit ops on 32-bit archs?

	// Split up named values into their components.
	// NOTE: the component values we are making are dead at this point.
	// We must do the opt pass before any deadcode elimination or we will
	// lose the name->value correspondence.
	for _, name := range f.Names {
		t := name.Type
		switch {
		case t.IsComplex():
			var elemType Type
			if t.Size() == 16 {
				elemType = f.Config.fe.TypeFloat64()
			} else {
				elemType = f.Config.fe.TypeFloat32()
			}
			rName := LocalSlot{name.N, elemType, name.Off}
			iName := LocalSlot{name.N, elemType, name.Off + elemType.Size()}
			f.Names = append(f.Names, rName, iName)
			for _, v := range f.NamedValues[name] {
				r := v.Block.NewValue1(v.Line, OpComplexReal, elemType, v)
				i := v.Block.NewValue1(v.Line, OpComplexImag, elemType, v)
				f.NamedValues[rName] = append(f.NamedValues[rName], r)
				f.NamedValues[iName] = append(f.NamedValues[iName], i)
			}
		case t.IsString():
			ptrType := f.Config.fe.TypeBytePtr()
			lenType := f.Config.fe.TypeInt()
			ptrName := LocalSlot{name.N, ptrType, name.Off}
			lenName := LocalSlot{name.N, lenType, name.Off + f.Config.PtrSize}
			f.Names = append(f.Names, ptrName, lenName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Line, OpStringPtr, ptrType, v)
				len := v.Block.NewValue1(v.Line, OpStringLen, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
			}
		case t.IsSlice():
			ptrType := f.Config.fe.TypeBytePtr()
			lenType := f.Config.fe.TypeInt()
			ptrName := LocalSlot{name.N, ptrType, name.Off}
			lenName := LocalSlot{name.N, lenType, name.Off + f.Config.PtrSize}
			capName := LocalSlot{name.N, lenType, name.Off + 2*f.Config.PtrSize}
			f.Names = append(f.Names, ptrName, lenName, capName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Line, OpSlicePtr, ptrType, v)
				len := v.Block.NewValue1(v.Line, OpSliceLen, lenType, v)
				cap := v.Block.NewValue1(v.Line, OpSliceCap, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
				f.NamedValues[capName] = append(f.NamedValues[capName], cap)
			}
		case t.IsInterface():
			ptrType := f.Config.fe.TypeBytePtr()
			typeName := LocalSlot{name.N, ptrType, name.Off}
			dataName := LocalSlot{name.N, ptrType, name.Off + f.Config.PtrSize}
			f.Names = append(f.Names, typeName, dataName)
			for _, v := range f.NamedValues[name] {
				typ := v.Block.NewValue1(v.Line, OpITab, ptrType, v)
				data := v.Block.NewValue1(v.Line, OpIData, ptrType, v)
				f.NamedValues[typeName] = append(f.NamedValues[typeName], typ)
				f.NamedValues[dataName] = append(f.NamedValues[dataName], data)
			}
			//case t.IsStruct():
			// TODO
		case t.Size() > f.Config.IntSize:
			f.Unimplementedf("undecomposed type %s", t)
		}
	}
}

func decomposeStringPhi(v *Value) {
	fe := v.Block.Func.Config.fe
	ptrType := fe.TypeBytePtr()
	lenType := fe.TypeInt()

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
	lenType := fe.TypeInt()

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

func decomposeComplexPhi(v *Value) {
	fe := v.Block.Func.Config.fe
	var partType Type
	switch z := v.Type.Size(); z {
	case 8:
		partType = fe.TypeFloat32()
	case 16:
		partType = fe.TypeFloat64()
	default:
		v.Fatalf("decomposeComplexPhi: bad complex size %d", z)
	}

	real := v.Block.NewValue0(v.Line, OpPhi, partType)
	imag := v.Block.NewValue0(v.Line, OpPhi, partType)
	for _, a := range v.Args {
		real.AddArg(a.Block.NewValue1(v.Line, OpComplexReal, partType, a))
		imag.AddArg(a.Block.NewValue1(v.Line, OpComplexImag, partType, a))
	}
	v.Op = OpComplexMake
	v.AuxInt = 0
	v.Aux = nil
	v.resetArgs()
	v.AddArg(real)
	v.AddArg(imag)
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
