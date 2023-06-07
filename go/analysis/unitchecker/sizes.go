// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unitchecker

import (
	"fmt"
	"go/types"
)

type gcSizes struct {
	WordSize int64 // word size in bytes - must be >= 4 (32bits)
	MaxAlign int64 // maximum alignment in bytes - must be >= 1
}

func (s *gcSizes) Alignof(T types.Type) int64 {
	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := T.Underlying().(type) {
	case *types.Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return s.Alignof(t.Elem())
	case *types.Struct:
		if t.NumFields() == 0 && isSyncAtomicAlign64(T) {
			// Special case: sync/atomic.align64 is an
			// empty struct we recognize as a signal that
			// the struct it contains must be
			// 64-bit-aligned.
			//
			// This logic is equivalent to the logic in
			// cmd/compile/internal/types/size.go:calcStructOffset
			return 8
		}

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
	case *types.Slice, *types.Interface:
		// Multiword data structures are effectively structs
		// in which each element has size PtrSize.
		return s.WordSize
	case *types.Basic:
		// Strings are like slices and interfaces.
		if t.Info()&types.IsString != 0 {
			return s.WordSize
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
	if a > s.MaxAlign {
		return s.MaxAlign
	}
	return a
}

func isComplex(T types.Type) bool {
	basic, ok := T.Underlying().(*types.Basic)
	return ok && basic.Info()&types.IsComplex != 0
}

func (s *gcSizes) Offsetsof(fields []*types.Var) []int64 {
	offsets := make([]int64, len(fields))
	var offs int64
	for i, f := range fields {
		if offs < 0 {
			// all remaining offsets are too large
			offsets[i] = -1
			continue
		}
		// offs >= 0
		typ := f.Type()
		a := s.Alignof(typ)
		offs = roundUp(offs, a) // possibly < 0 if align overflows
		offsets[i] = offs
		if d := s.Sizeof(typ); d >= 0 && offs >= 0 {
			offs += d // ok to overflow to < 0
		} else {
			offs = -1
		}
	}
	return offsets
}

func (s *gcSizes) Sizeof(T types.Type) int64 {
	switch t := T.Underlying().(type) {
	case *types.Basic:
		k := t.Kind()
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		switch k {
		case types.String:
			return s.WordSize * 2
		case types.Int, types.Uint, types.Uintptr, types.UnsafePointer:
			return s.WordSize
		}
		panic(fmt.Sprintf("unimplemented basic: %v (kind %v)", T, k))
	case *types.Array:
		n := t.Len()
		if n <= 0 {
			return 0
		}
		// n > 0
		// gc: Size includes alignment padding.
		esize := s.Sizeof(t.Elem())
		if esize < 0 {
			return -1 // array element too large
		}
		if esize == 0 {
			return 0 // 0-size element
		}
		// esize > 0
		// Final size is esize * n; and size must be <= maxInt64.
		const maxInt64 = 1<<63 - 1
		if esize > maxInt64/n {
			return -1 // esize * n overflows
		}
		return esize * n
	case *types.Slice:
		return s.WordSize * 3
	case *types.Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		// n > 0
		fields := make([]*types.Var, n)
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
		return roundUp(offsets[n-1]+last, s.Alignof(t)) // may overflow to < 0 which is ok
	case *types.Interface:
		return s.WordSize * 2
	case *types.Chan, *types.Map, *types.Pointer, *types.Signature:
		return s.WordSize
	default:
		panic(fmt.Sprintf("Sizeof(%T) unimplemented", t))
	}
}

func isSyncAtomicAlign64(T types.Type) bool {
	named, ok := T.(*types.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	return obj.Name() == "align64" &&
		obj.Pkg() != nil &&
		(obj.Pkg().Path() == "sync/atomic" ||
			obj.Pkg().Path() == "runtime/internal/atomic")
}

// roundUp rounds o to a multiple of r, r is a power of 2.
func roundUp(o int64, r int64) int64 {
	if r < 1 || r > 8 || r&(r-1) != 0 {
		panic(fmt.Sprintf("Round %d", r))
	}
	return (o + r - 1) &^ (r - 1)
}

var basicSizes = [...]byte{
	types.Invalid:    1,
	types.Bool:       1,
	types.Int8:       1,
	types.Int16:      2,
	types.Int32:      4,
	types.Int64:      8,
	types.Uint8:      1,
	types.Uint16:     2,
	types.Uint32:     4,
	types.Uint64:     8,
	types.Float32:    4,
	types.Float64:    8,
	types.Complex64:  8,
	types.Complex128: 16,
}

// common architecture word sizes and alignments
var gcArchSizes = map[string]*gcSizes{
	"386":      {4, 4},
	"amd64":    {8, 8},
	"amd64p32": {4, 8},
	"arm":      {4, 4},
	"arm64":    {8, 8},
	"loong64":  {8, 8},
	"mips":     {4, 4},
	"mipsle":   {4, 4},
	"mips64":   {8, 8},
	"mips64le": {8, 8},
	"ppc64":    {8, 8},
	"ppc64le":  {8, 8},
	"riscv64":  {8, 8},
	"s390x":    {8, 8},
	"sparc64":  {8, 8},
	"wasm":     {8, 8},
	// When adding more architectures here,
	// update the doc string of sizesFor below.
}

// sizesFor returns the go/types.Sizes used by a compiler for an architecture.
// The result is nil if a compiler/architecture pair is not known.
func sizesFor(compiler, arch string) types.Sizes {
	if compiler != "gc" {
		return nil
	}
	s, ok := gcArchSizes[arch]
	if !ok {
		return nil
	}
	return s
}
