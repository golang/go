// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// decompose converts phi ops on compound builtin types into phi
// ops on simple types.
// (The remaining compound ops are decomposed with rewrite rules.)
func decomposeBuiltIn(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			decomposeBuiltInPhi(v)
		}
	}

	// Split up named values into their components.
	// NOTE: the component values we are making are dead at this point.
	// We must do the opt pass before any deadcode elimination or we will
	// lose the name->value correspondence.
	var newNames []LocalSlot
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
			rName, iName := f.Config.fe.SplitComplex(name)
			newNames = append(newNames, rName, iName)
			for _, v := range f.NamedValues[name] {
				r := v.Block.NewValue1(v.Line, OpComplexReal, elemType, v)
				i := v.Block.NewValue1(v.Line, OpComplexImag, elemType, v)
				f.NamedValues[rName] = append(f.NamedValues[rName], r)
				f.NamedValues[iName] = append(f.NamedValues[iName], i)
			}
			delete(f.NamedValues, name)
		case t.IsString():
			ptrType := f.Config.fe.TypeBytePtr()
			lenType := f.Config.fe.TypeInt()
			ptrName, lenName := f.Config.fe.SplitString(name)
			newNames = append(newNames, ptrName, lenName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Line, OpStringPtr, ptrType, v)
				len := v.Block.NewValue1(v.Line, OpStringLen, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
			}
			delete(f.NamedValues, name)
		case t.IsSlice():
			ptrType := f.Config.fe.TypeBytePtr()
			lenType := f.Config.fe.TypeInt()
			ptrName, lenName, capName := f.Config.fe.SplitSlice(name)
			newNames = append(newNames, ptrName, lenName, capName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Line, OpSlicePtr, ptrType, v)
				len := v.Block.NewValue1(v.Line, OpSliceLen, lenType, v)
				cap := v.Block.NewValue1(v.Line, OpSliceCap, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
				f.NamedValues[capName] = append(f.NamedValues[capName], cap)
			}
			delete(f.NamedValues, name)
		case t.IsInterface():
			ptrType := f.Config.fe.TypeBytePtr()
			typeName, dataName := f.Config.fe.SplitInterface(name)
			newNames = append(newNames, typeName, dataName)
			for _, v := range f.NamedValues[name] {
				typ := v.Block.NewValue1(v.Line, OpITab, ptrType, v)
				data := v.Block.NewValue1(v.Line, OpIData, ptrType, v)
				f.NamedValues[typeName] = append(f.NamedValues[typeName], typ)
				f.NamedValues[dataName] = append(f.NamedValues[dataName], data)
			}
			delete(f.NamedValues, name)
		case t.Size() > f.Config.IntSize:
			f.Unimplementedf("undecomposed named type %s %s", name, t)
		default:
			newNames = append(newNames, name)
		}
	}
	f.Names = newNames
}

func decomposeBuiltInPhi(v *Value) {
	// TODO: decompose 64-bit ops on 32-bit archs?
	switch {
	case v.Type.IsComplex():
		decomposeComplexPhi(v)
	case v.Type.IsString():
		decomposeStringPhi(v)
	case v.Type.IsSlice():
		decomposeSlicePhi(v)
	case v.Type.IsInterface():
		decomposeInterfacePhi(v)
	case v.Type.Size() > v.Block.Func.Config.IntSize:
		v.Unimplementedf("undecomposed type %s", v.Type)
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
	v.reset(OpStringMake)
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
	v.reset(OpSliceMake)
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
	v.reset(OpComplexMake)
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
	v.reset(OpIMake)
	v.AddArg(itab)
	v.AddArg(data)
}

func decomposeUser(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			decomposeUserPhi(v)
		}
	}
	// Split up named values into their components.
	// NOTE: the component values we are making are dead at this point.
	// We must do the opt pass before any deadcode elimination or we will
	// lose the name->value correspondence.
	i := 0
	var fnames []LocalSlot
	var newNames []LocalSlot
	for _, name := range f.Names {
		t := name.Type
		switch {
		case t.IsStruct():
			n := t.NumFields()
			fnames = fnames[:0]
			for i := 0; i < n; i++ {
				fnames = append(fnames, f.Config.fe.SplitStruct(name, i))
			}
			for _, v := range f.NamedValues[name] {
				for i := 0; i < n; i++ {
					x := v.Block.NewValue1I(v.Line, OpStructSelect, t.FieldType(i), int64(i), v)
					f.NamedValues[fnames[i]] = append(f.NamedValues[fnames[i]], x)
				}
			}
			delete(f.NamedValues, name)
			newNames = append(newNames, fnames...)
		default:
			f.Names[i] = name
			i++
		}
	}
	f.Names = f.Names[:i]
	f.Names = append(f.Names, newNames...)
}

func decomposeUserPhi(v *Value) {
	switch {
	case v.Type.IsStruct():
		decomposeStructPhi(v)
	}
	// TODO: Arrays of length 1?
}

func decomposeStructPhi(v *Value) {
	t := v.Type
	n := t.NumFields()
	var fields [MaxStruct]*Value
	for i := 0; i < n; i++ {
		fields[i] = v.Block.NewValue0(v.Line, OpPhi, t.FieldType(i))
	}
	for _, a := range v.Args {
		for i := 0; i < n; i++ {
			fields[i].AddArg(a.Block.NewValue1I(v.Line, OpStructSelect, t.FieldType(i), int64(i), a))
		}
	}
	v.reset(StructMakeOp(n))
	v.AddArgs(fields[:n]...)

	// Recursively decompose phis for each field.
	for _, f := range fields[:n] {
		if f.Type.IsStruct() {
			decomposeStructPhi(f)
		}
	}
}

// MaxStruct is the maximum number of fields a struct
// can have and still be SSAable.
const MaxStruct = 4

// StructMakeOp returns the opcode to construct a struct with the
// given number of fields.
func StructMakeOp(nf int) Op {
	switch nf {
	case 0:
		return OpStructMake0
	case 1:
		return OpStructMake1
	case 2:
		return OpStructMake2
	case 3:
		return OpStructMake3
	case 4:
		return OpStructMake4
	}
	panic("too many fields in an SSAable struct")
}
