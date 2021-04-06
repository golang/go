// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

// Code below based on go/types.StdSizes.
// Intentional differences are marked with "gc:".

type gcSizes struct{}

func (s *gcSizes) Alignof(T types2.Type) int64 {
	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := T.Underlying().(type) {
	case *types2.Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return s.Alignof(t.Elem())
	case *types2.Struct:
		// spec: "For a variable x of struct type: unsafe.Alignof(x)
		// is the largest of the values unsafe.Alignof(x.f) for each
		// field f of x, but at least 1."
		max := int64(1)
		for i, nf := 0, t.NumFields(); i < nf; i++ {
			if a := s.Alignof(t.Field(i).Type()); a > max {
				max = a
			}
		}
		return max
	case *types2.Slice, *types2.Interface:
		// Multiword data structures are effectively structs
		// in which each element has size PtrSize.
		return int64(types.PtrSize)
	case *types2.Basic:
		// Strings are like slices and interfaces.
		if t.Info()&types2.IsString != 0 {
			return int64(types.PtrSize)
		}
	}
	a := s.Sizeof(T) // may be 0
	// spec: "For a variable x of any type: unsafe.Alignof(x) is at least 1."
	if a < 1 {
		return 1
	}
	// complex{64,128} are aligned like [2]float{32,64}.
	if isComplex(T) {
		a /= 2
	}
	if a > int64(types.RegSize) {
		return int64(types.RegSize)
	}
	return a
}

func isComplex(T types2.Type) bool {
	basic, ok := T.Underlying().(*types2.Basic)
	return ok && basic.Info()&types2.IsComplex != 0
}

func (s *gcSizes) Offsetsof(fields []*types2.Var) []int64 {
	offsets := make([]int64, len(fields))
	var o int64
	for i, f := range fields {
		typ := f.Type()
		a := s.Alignof(typ)
		o = types.Rnd(o, a)
		offsets[i] = o
		o += s.Sizeof(typ)
	}
	return offsets
}

func (s *gcSizes) Sizeof(T types2.Type) int64 {
	switch t := T.Underlying().(type) {
	case *types2.Basic:
		k := t.Kind()
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		switch k {
		case types2.String:
			return int64(types.PtrSize) * 2
		case types2.Int, types2.Uint, types2.Uintptr, types2.UnsafePointer:
			return int64(types.PtrSize)
		}
		panic(fmt.Sprintf("unimplemented basic: %v (kind %v)", T, k))
	case *types2.Array:
		n := t.Len()
		if n <= 0 {
			return 0
		}
		// n > 0
		// gc: Size includes alignment padding.
		return s.Sizeof(t.Elem()) * n
	case *types2.Slice:
		return int64(types.PtrSize) * 3
	case *types2.Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		fields := make([]*types2.Var, n)
		for i := range fields {
			fields[i] = t.Field(i)
		}
		offsets := s.Offsetsof(fields)

		// gc: The last field of a non-zero-sized struct is not allowed to
		// have size 0.
		last := s.Sizeof(fields[n-1].Type())
		if last == 0 && offsets[n-1] > 0 {
			last = 1
		}

		// gc: Size includes alignment padding.
		return types.Rnd(offsets[n-1]+last, s.Alignof(t))
	case *types2.Interface:
		return int64(types.PtrSize) * 2
	case *types2.Chan, *types2.Map, *types2.Pointer, *types2.Signature:
		return int64(types.PtrSize)
	default:
		panic(fmt.Sprintf("unimplemented type: %T", t))
	}
}

var basicSizes = [...]byte{
	types2.Bool:       1,
	types2.Int8:       1,
	types2.Int16:      2,
	types2.Int32:      4,
	types2.Int64:      8,
	types2.Uint8:      1,
	types2.Uint16:     2,
	types2.Uint32:     4,
	types2.Uint64:     8,
	types2.Float32:    4,
	types2.Float64:    8,
	types2.Complex64:  8,
	types2.Complex128: 16,
}
