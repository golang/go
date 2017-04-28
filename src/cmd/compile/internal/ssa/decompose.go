// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/types"

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
		case t.IsInteger() && t.Size() > f.Config.RegSize:
			var elemType *types.Type
			if t.IsSigned() {
				elemType = f.Config.Types.Int32
			} else {
				elemType = f.Config.Types.UInt32
			}
			hiName, loName := f.fe.SplitInt64(name)
			newNames = append(newNames, hiName, loName)
			for _, v := range f.NamedValues[name] {
				hi := v.Block.NewValue1(v.Pos, OpInt64Hi, elemType, v)
				lo := v.Block.NewValue1(v.Pos, OpInt64Lo, f.Config.Types.UInt32, v)
				f.NamedValues[hiName] = append(f.NamedValues[hiName], hi)
				f.NamedValues[loName] = append(f.NamedValues[loName], lo)
			}
			delete(f.NamedValues, name)
		case t.IsComplex():
			var elemType *types.Type
			if t.Size() == 16 {
				elemType = f.Config.Types.Float64
			} else {
				elemType = f.Config.Types.Float32
			}
			rName, iName := f.fe.SplitComplex(name)
			newNames = append(newNames, rName, iName)
			for _, v := range f.NamedValues[name] {
				r := v.Block.NewValue1(v.Pos, OpComplexReal, elemType, v)
				i := v.Block.NewValue1(v.Pos, OpComplexImag, elemType, v)
				f.NamedValues[rName] = append(f.NamedValues[rName], r)
				f.NamedValues[iName] = append(f.NamedValues[iName], i)
			}
			delete(f.NamedValues, name)
		case t.IsString():
			ptrType := f.Config.Types.BytePtr
			lenType := f.Config.Types.Int
			ptrName, lenName := f.fe.SplitString(name)
			newNames = append(newNames, ptrName, lenName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Pos, OpStringPtr, ptrType, v)
				len := v.Block.NewValue1(v.Pos, OpStringLen, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
			}
			delete(f.NamedValues, name)
		case t.IsSlice():
			ptrType := f.Config.Types.BytePtr
			lenType := f.Config.Types.Int
			ptrName, lenName, capName := f.fe.SplitSlice(name)
			newNames = append(newNames, ptrName, lenName, capName)
			for _, v := range f.NamedValues[name] {
				ptr := v.Block.NewValue1(v.Pos, OpSlicePtr, ptrType, v)
				len := v.Block.NewValue1(v.Pos, OpSliceLen, lenType, v)
				cap := v.Block.NewValue1(v.Pos, OpSliceCap, lenType, v)
				f.NamedValues[ptrName] = append(f.NamedValues[ptrName], ptr)
				f.NamedValues[lenName] = append(f.NamedValues[lenName], len)
				f.NamedValues[capName] = append(f.NamedValues[capName], cap)
			}
			delete(f.NamedValues, name)
		case t.IsInterface():
			ptrType := f.Config.Types.BytePtr
			typeName, dataName := f.fe.SplitInterface(name)
			newNames = append(newNames, typeName, dataName)
			for _, v := range f.NamedValues[name] {
				typ := v.Block.NewValue1(v.Pos, OpITab, ptrType, v)
				data := v.Block.NewValue1(v.Pos, OpIData, ptrType, v)
				f.NamedValues[typeName] = append(f.NamedValues[typeName], typ)
				f.NamedValues[dataName] = append(f.NamedValues[dataName], data)
			}
			delete(f.NamedValues, name)
		case t.IsFloat():
			// floats are never decomposed, even ones bigger than RegSize
		case t.Size() > f.Config.RegSize:
			f.Fatalf("undecomposed named type %v %v", name, t)
		default:
			newNames = append(newNames, name)
		}
	}
	f.Names = newNames
}

func decomposeBuiltInPhi(v *Value) {
	switch {
	case v.Type.IsInteger() && v.Type.Size() > v.Block.Func.Config.RegSize:
		decomposeInt64Phi(v)
	case v.Type.IsComplex():
		decomposeComplexPhi(v)
	case v.Type.IsString():
		decomposeStringPhi(v)
	case v.Type.IsSlice():
		decomposeSlicePhi(v)
	case v.Type.IsInterface():
		decomposeInterfacePhi(v)
	case v.Type.IsFloat():
		// floats are never decomposed, even ones bigger than RegSize
	case v.Type.Size() > v.Block.Func.Config.RegSize:
		v.Fatalf("undecomposed type %s", v.Type)
	}
}

func decomposeStringPhi(v *Value) {
	types := &v.Block.Func.Config.Types
	ptrType := types.BytePtr
	lenType := types.Int

	ptr := v.Block.NewValue0(v.Pos, OpPhi, ptrType)
	len := v.Block.NewValue0(v.Pos, OpPhi, lenType)
	for _, a := range v.Args {
		ptr.AddArg(a.Block.NewValue1(v.Pos, OpStringPtr, ptrType, a))
		len.AddArg(a.Block.NewValue1(v.Pos, OpStringLen, lenType, a))
	}
	v.reset(OpStringMake)
	v.AddArg(ptr)
	v.AddArg(len)
}

func decomposeSlicePhi(v *Value) {
	types := &v.Block.Func.Config.Types
	ptrType := types.BytePtr
	lenType := types.Int

	ptr := v.Block.NewValue0(v.Pos, OpPhi, ptrType)
	len := v.Block.NewValue0(v.Pos, OpPhi, lenType)
	cap := v.Block.NewValue0(v.Pos, OpPhi, lenType)
	for _, a := range v.Args {
		ptr.AddArg(a.Block.NewValue1(v.Pos, OpSlicePtr, ptrType, a))
		len.AddArg(a.Block.NewValue1(v.Pos, OpSliceLen, lenType, a))
		cap.AddArg(a.Block.NewValue1(v.Pos, OpSliceCap, lenType, a))
	}
	v.reset(OpSliceMake)
	v.AddArg(ptr)
	v.AddArg(len)
	v.AddArg(cap)
}

func decomposeInt64Phi(v *Value) {
	cfgtypes := &v.Block.Func.Config.Types
	var partType *types.Type
	if v.Type.IsSigned() {
		partType = cfgtypes.Int32
	} else {
		partType = cfgtypes.UInt32
	}

	hi := v.Block.NewValue0(v.Pos, OpPhi, partType)
	lo := v.Block.NewValue0(v.Pos, OpPhi, cfgtypes.UInt32)
	for _, a := range v.Args {
		hi.AddArg(a.Block.NewValue1(v.Pos, OpInt64Hi, partType, a))
		lo.AddArg(a.Block.NewValue1(v.Pos, OpInt64Lo, cfgtypes.UInt32, a))
	}
	v.reset(OpInt64Make)
	v.AddArg(hi)
	v.AddArg(lo)
}

func decomposeComplexPhi(v *Value) {
	cfgtypes := &v.Block.Func.Config.Types
	var partType *types.Type
	switch z := v.Type.Size(); z {
	case 8:
		partType = cfgtypes.Float32
	case 16:
		partType = cfgtypes.Float64
	default:
		v.Fatalf("decomposeComplexPhi: bad complex size %d", z)
	}

	real := v.Block.NewValue0(v.Pos, OpPhi, partType)
	imag := v.Block.NewValue0(v.Pos, OpPhi, partType)
	for _, a := range v.Args {
		real.AddArg(a.Block.NewValue1(v.Pos, OpComplexReal, partType, a))
		imag.AddArg(a.Block.NewValue1(v.Pos, OpComplexImag, partType, a))
	}
	v.reset(OpComplexMake)
	v.AddArg(real)
	v.AddArg(imag)
}

func decomposeInterfacePhi(v *Value) {
	ptrType := v.Block.Func.Config.Types.BytePtr

	itab := v.Block.NewValue0(v.Pos, OpPhi, ptrType)
	data := v.Block.NewValue0(v.Pos, OpPhi, ptrType)
	for _, a := range v.Args {
		itab.AddArg(a.Block.NewValue1(v.Pos, OpITab, ptrType, a))
		data.AddArg(a.Block.NewValue1(v.Pos, OpIData, ptrType, a))
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
				fnames = append(fnames, f.fe.SplitStruct(name, i))
			}
			for _, v := range f.NamedValues[name] {
				for i := 0; i < n; i++ {
					x := v.Block.NewValue1I(v.Pos, OpStructSelect, t.FieldType(i), int64(i), v)
					f.NamedValues[fnames[i]] = append(f.NamedValues[fnames[i]], x)
				}
			}
			delete(f.NamedValues, name)
			newNames = append(newNames, fnames...)
		case t.IsArray():
			if t.NumElem() == 0 {
				// TODO(khr): Not sure what to do here.  Probably nothing.
				// Names for empty arrays aren't important.
				break
			}
			if t.NumElem() != 1 {
				f.Fatalf("array not of size 1")
			}
			elemName := f.fe.SplitArray(name)
			for _, v := range f.NamedValues[name] {
				e := v.Block.NewValue1I(v.Pos, OpArraySelect, t.ElemType(), 0, v)
				f.NamedValues[elemName] = append(f.NamedValues[elemName], e)
			}

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
	case v.Type.IsArray():
		decomposeArrayPhi(v)
	}
}

// decomposeStructPhi replaces phi-of-struct with structmake(phi-for-each-field),
// and then recursively decomposes the phis for each field.
func decomposeStructPhi(v *Value) {
	t := v.Type
	n := t.NumFields()
	var fields [MaxStruct]*Value
	for i := 0; i < n; i++ {
		fields[i] = v.Block.NewValue0(v.Pos, OpPhi, t.FieldType(i))
	}
	for _, a := range v.Args {
		for i := 0; i < n; i++ {
			fields[i].AddArg(a.Block.NewValue1I(v.Pos, OpStructSelect, t.FieldType(i), int64(i), a))
		}
	}
	v.reset(StructMakeOp(n))
	v.AddArgs(fields[:n]...)

	// Recursively decompose phis for each field.
	for _, f := range fields[:n] {
		decomposeUserPhi(f)
	}
}

// decomposeArrayPhi replaces phi-of-array with arraymake(phi-of-array-element),
// and then recursively decomposes the element phi.
func decomposeArrayPhi(v *Value) {
	t := v.Type
	if t.NumElem() == 0 {
		v.reset(OpArrayMake0)
		return
	}
	if t.NumElem() != 1 {
		v.Fatalf("SSAable array must have no more than 1 element")
	}
	elem := v.Block.NewValue0(v.Pos, OpPhi, t.ElemType())
	for _, a := range v.Args {
		elem.AddArg(a.Block.NewValue1I(v.Pos, OpArraySelect, t.ElemType(), 0, a))
	}
	v.reset(OpArrayMake1)
	v.AddArg(elem)

	// Recursively decompose elem phi.
	decomposeUserPhi(elem)
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
