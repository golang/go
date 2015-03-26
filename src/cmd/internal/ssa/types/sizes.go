// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Sizes.

package types

import "log"

// Sizes defines the sizing functions for package unsafe.
type Sizes interface {
	// Alignof returns the alignment of a variable of type T.
	// Alignof must implement the alignment guarantees required by the spec.
	Alignof(T Type) int64

	// Offsetsof returns the offsets of the given struct fields, in bytes.
	// Offsetsof must implement the offset guarantees required by the spec.
	Offsetsof(fields []*Var) []int64

	// Sizeof returns the size of a variable of type T.
	// Sizeof must implement the size guarantees required by the spec.
	Sizeof(T Type) int64
}

// StdSizes is a convenience type for creating commonly used Sizes.
// It makes the following simplifying assumptions:
//
//	- The size of explicitly sized basic types (int16, etc.) is the
//	  specified size.
//	- The size of strings and interfaces is 2*WordSize.
//	- The size of slices is 3*WordSize.
//	- The size of an array of n elements corresponds to the size of
//	  a struct of n consecutive fields of the array's element type.
//      - The size of a struct is the offset of the last field plus that
//	  field's size. As with all element types, if the struct is used
//	  in an array its size must first be aligned to a multiple of the
//	  struct's alignment.
//	- All other types have size WordSize.
//	- Arrays and structs are aligned per spec definition; all other
//	  types are naturally aligned with a maximum alignment MaxAlign.
//
// *StdSizes implements Sizes.
//
type StdSizes struct {
	WordSize int64 // word size in bytes - must be >= 4 (32bits)
	MaxAlign int64 // maximum alignment in bytes - must be >= 1
}

func (s *StdSizes) Alignof(T Type) int64 {
	a := s.Sizeof(T) // may be 0
	// spec: "For a variable x of any type: unsafe.Alignof(x) is at least 1."
	if a < 1 {
		return 1
	}
	if a > s.MaxAlign {
		return s.MaxAlign
	}
	return a
}

func (s *StdSizes) Offsetsof(fields []*Var) []int64 {
	offsets := make([]int64, len(fields))
	var o int64
	for i, f := range fields {
		a := s.Alignof(f.typ)
		o = align(o, a)
		offsets[i] = o
		o += s.Sizeof(f.typ)
	}
	return offsets
}

var basicSizes = [...]byte{
	Bool:       1,
	Int8:       1,
	Int16:      2,
	Int32:      4,
	Int64:      8,
	Uint8:      1,
	Uint16:     2,
	Uint32:     4,
	Uint64:     8,
	Float32:    4,
	Float64:    8,
	Complex64:  8,
	Complex128: 16,
}

func (s *StdSizes) Sizeof(T Type) int64 {
	switch t := T.Underlying().(type) {
	case *Basic:
		k := t.kind
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		if k == String {
			return s.WordSize * 2
		}
	case *Slice:
		return s.WordSize * 3
	default:
		log.Fatalf("not implemented")
	}
	return s.WordSize // catch-all
}

// stdSizes is used if Config.Sizes == nil.
var stdSizes = StdSizes{8, 8}

// align returns the smallest y >= x such that y % a == 0.
func align(x, a int64) int64 {
	y := x + a - 1
	return y - y%a
}
