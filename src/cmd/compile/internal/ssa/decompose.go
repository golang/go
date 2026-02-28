// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"sort"
)

// decompose converts phi ops on compound builtin types into phi
// ops on simple types, then invokes rewrite rules to decompose
// other ops on those types.
func decomposeBuiltIn(f *Func) {
	// Decompose phis
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			decomposeBuiltInPhi(v)
		}
	}

	// Decompose other values
	// Note: Leave dead values because we need to keep the original
	// values around so the name component resolution below can still work.
	applyRewrite(f, rewriteBlockdec, rewriteValuedec, leaveDeadValues)
	if f.Config.RegSize == 4 {
		applyRewrite(f, rewriteBlockdec64, rewriteValuedec64, leaveDeadValues)
	}

	// Split up named values into their components.
	// accumulate old names for aggregates (that are decomposed) in toDelete for efficient bulk deletion,
	// accumulate new LocalSlots in newNames for addition after the iteration.  This decomposition is for
	// builtin types with leaf components, and thus there is no need to reprocess the newly create LocalSlots.
	var toDelete []namedVal
	var newNames []*LocalSlot
	for i, name := range f.Names {
		t := name.Type
		switch {
		case t.IsInteger() && t.Size() > f.Config.RegSize:
			hiName, loName := f.SplitInt64(name)
			newNames = maybeAppend2(f, newNames, hiName, loName)
			for j, v := range f.NamedValues[*name] {
				if v.Op != OpInt64Make {
					continue
				}
				f.NamedValues[*hiName] = append(f.NamedValues[*hiName], v.Args[0])
				f.NamedValues[*loName] = append(f.NamedValues[*loName], v.Args[1])
				toDelete = append(toDelete, namedVal{i, j})
			}
		case t.IsComplex():
			rName, iName := f.SplitComplex(name)
			newNames = maybeAppend2(f, newNames, rName, iName)
			for j, v := range f.NamedValues[*name] {
				if v.Op != OpComplexMake {
					continue
				}
				f.NamedValues[*rName] = append(f.NamedValues[*rName], v.Args[0])
				f.NamedValues[*iName] = append(f.NamedValues[*iName], v.Args[1])
				toDelete = append(toDelete, namedVal{i, j})
			}
		case t.IsString():
			ptrName, lenName := f.SplitString(name)
			newNames = maybeAppend2(f, newNames, ptrName, lenName)
			for j, v := range f.NamedValues[*name] {
				if v.Op != OpStringMake {
					continue
				}
				f.NamedValues[*ptrName] = append(f.NamedValues[*ptrName], v.Args[0])
				f.NamedValues[*lenName] = append(f.NamedValues[*lenName], v.Args[1])
				toDelete = append(toDelete, namedVal{i, j})
			}
		case t.IsSlice():
			ptrName, lenName, capName := f.SplitSlice(name)
			newNames = maybeAppend2(f, newNames, ptrName, lenName)
			newNames = maybeAppend(f, newNames, capName)
			for j, v := range f.NamedValues[*name] {
				if v.Op != OpSliceMake {
					continue
				}
				f.NamedValues[*ptrName] = append(f.NamedValues[*ptrName], v.Args[0])
				f.NamedValues[*lenName] = append(f.NamedValues[*lenName], v.Args[1])
				f.NamedValues[*capName] = append(f.NamedValues[*capName], v.Args[2])
				toDelete = append(toDelete, namedVal{i, j})
			}
		case t.IsInterface():
			typeName, dataName := f.SplitInterface(name)
			newNames = maybeAppend2(f, newNames, typeName, dataName)
			for j, v := range f.NamedValues[*name] {
				if v.Op != OpIMake {
					continue
				}
				f.NamedValues[*typeName] = append(f.NamedValues[*typeName], v.Args[0])
				f.NamedValues[*dataName] = append(f.NamedValues[*dataName], v.Args[1])
				toDelete = append(toDelete, namedVal{i, j})
			}
		case t.IsFloat():
			// floats are never decomposed, even ones bigger than RegSize
		case t.Size() > f.Config.RegSize:
			f.Fatalf("undecomposed named type %s %v", name, t)
		}
	}

	deleteNamedVals(f, toDelete)
	f.Names = append(f.Names, newNames...)
}

func maybeAppend(f *Func, ss []*LocalSlot, s *LocalSlot) []*LocalSlot {
	if _, ok := f.NamedValues[*s]; !ok {
		f.NamedValues[*s] = nil
		return append(ss, s)
	}
	return ss
}

func maybeAppend2(f *Func, ss []*LocalSlot, s1, s2 *LocalSlot) []*LocalSlot {
	return maybeAppend(f, maybeAppend(f, ss, s1), s2)
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
	ptrType := v.Type.Elem().PtrTo()
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
	uintptrType := v.Block.Func.Config.Types.Uintptr
	ptrType := v.Block.Func.Config.Types.BytePtr

	itab := v.Block.NewValue0(v.Pos, OpPhi, uintptrType)
	data := v.Block.NewValue0(v.Pos, OpPhi, ptrType)
	for _, a := range v.Args {
		itab.AddArg(a.Block.NewValue1(v.Pos, OpITab, uintptrType, a))
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
	i := 0
	var newNames []*LocalSlot
	for _, name := range f.Names {
		t := name.Type
		switch {
		case t.IsStruct():
			newNames = decomposeUserStructInto(f, name, newNames)
		case t.IsArray():
			newNames = decomposeUserArrayInto(f, name, newNames)
		default:
			f.Names[i] = name
			i++
		}
	}
	f.Names = f.Names[:i]
	f.Names = append(f.Names, newNames...)
}

// decomposeUserArrayInto creates names for the element(s) of arrays referenced
// by name where possible, and appends those new names to slots, which is then
// returned.
func decomposeUserArrayInto(f *Func, name *LocalSlot, slots []*LocalSlot) []*LocalSlot {
	t := name.Type
	if t.NumElem() == 0 {
		// TODO(khr): Not sure what to do here.  Probably nothing.
		// Names for empty arrays aren't important.
		return slots
	}
	if t.NumElem() != 1 {
		// shouldn't get here due to CanSSA
		f.Fatalf("array not of size 1")
	}
	elemName := f.SplitArray(name)
	var keep []*Value
	for _, v := range f.NamedValues[*name] {
		if v.Op != OpArrayMake1 {
			keep = append(keep, v)
			continue
		}
		f.NamedValues[*elemName] = append(f.NamedValues[*elemName], v.Args[0])
	}
	if len(keep) == 0 {
		// delete the name for the array as a whole
		delete(f.NamedValues, *name)
	} else {
		f.NamedValues[*name] = keep
	}

	if t.Elem().IsArray() {
		return decomposeUserArrayInto(f, elemName, slots)
	} else if t.Elem().IsStruct() {
		return decomposeUserStructInto(f, elemName, slots)
	}

	return append(slots, elemName)
}

// decomposeUserStructInto creates names for the fields(s) of structs referenced
// by name where possible, and appends those new names to slots, which is then
// returned.
func decomposeUserStructInto(f *Func, name *LocalSlot, slots []*LocalSlot) []*LocalSlot {
	fnames := []*LocalSlot{} // slots for struct in name
	t := name.Type
	n := t.NumFields()

	for i := 0; i < n; i++ {
		fs := f.SplitStruct(name, i)
		fnames = append(fnames, fs)
		// arrays and structs will be decomposed further, so
		// there's no need to record a name
		if !fs.Type.IsArray() && !fs.Type.IsStruct() {
			slots = maybeAppend(f, slots, fs)
		}
	}

	makeOp := StructMakeOp(n)
	var keep []*Value
	// create named values for each struct field
	for _, v := range f.NamedValues[*name] {
		if v.Op != makeOp {
			keep = append(keep, v)
			continue
		}
		for i := 0; i < len(fnames); i++ {
			f.NamedValues[*fnames[i]] = append(f.NamedValues[*fnames[i]], v.Args[i])
		}
	}
	if len(keep) == 0 {
		// delete the name for the struct as a whole
		delete(f.NamedValues, *name)
	} else {
		f.NamedValues[*name] = keep
	}

	// now that this f.NamedValues contains values for the struct
	// fields, recurse into nested structs
	for i := 0; i < n; i++ {
		if name.Type.FieldType(i).IsStruct() {
			slots = decomposeUserStructInto(f, fnames[i], slots)
			delete(f.NamedValues, *fnames[i])
		} else if name.Type.FieldType(i).IsArray() {
			slots = decomposeUserArrayInto(f, fnames[i], slots)
			delete(f.NamedValues, *fnames[i])
		}
	}
	return slots
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
	elem := v.Block.NewValue0(v.Pos, OpPhi, t.Elem())
	for _, a := range v.Args {
		elem.AddArg(a.Block.NewValue1I(v.Pos, OpArraySelect, t.Elem(), 0, a))
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

type namedVal struct {
	locIndex, valIndex int // f.NamedValues[f.Names[locIndex]][valIndex] = key
}

// deleteNamedVals removes particular values with debugger names from f's naming data structures,
// removes all values with OpInvalid, and re-sorts the list of Names.
func deleteNamedVals(f *Func, toDelete []namedVal) {
	// Arrange to delete from larger indices to smaller, to ensure swap-with-end deletion does not invalidate pending indices.
	sort.Slice(toDelete, func(i, j int) bool {
		if toDelete[i].locIndex != toDelete[j].locIndex {
			return toDelete[i].locIndex > toDelete[j].locIndex
		}
		return toDelete[i].valIndex > toDelete[j].valIndex

	})

	// Get rid of obsolete names
	for _, d := range toDelete {
		loc := f.Names[d.locIndex]
		vals := f.NamedValues[*loc]
		l := len(vals) - 1
		if l > 0 {
			vals[d.valIndex] = vals[l]
		}
		vals[l] = nil
		f.NamedValues[*loc] = vals[:l]
	}
	// Delete locations with no values attached.
	end := len(f.Names)
	for i := len(f.Names) - 1; i >= 0; i-- {
		loc := f.Names[i]
		vals := f.NamedValues[*loc]
		last := len(vals)
		for j := len(vals) - 1; j >= 0; j-- {
			if vals[j].Op == OpInvalid {
				last--
				vals[j] = vals[last]
				vals[last] = nil
			}
		}
		if last < len(vals) {
			f.NamedValues[*loc] = vals[:last]
		}
		if len(vals) == 0 {
			delete(f.NamedValues, *loc)
			end--
			f.Names[i] = f.Names[end]
			f.Names[end] = nil
		}
	}
	f.Names = f.Names[:end]
}
