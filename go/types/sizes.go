// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Sizes.

package types

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
	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := T.Underlying().(type) {
	case *Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return s.Alignof(t.elem)
	case *Struct:
		// spec: "For a variable x of struct type: unsafe.Alignof(x)
		// is the largest of the values unsafe.Alignof(x.f) for each
		// field f of x, but at least 1."
		max := int64(1)
		for _, f := range t.fields {
			if a := s.Alignof(f.typ); a > max {
				max = a
			}
		}
		return max
	}
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
		assert(isTyped(T))
		k := t.kind
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		if k == String {
			return s.WordSize * 2
		}
	case *Array:
		n := t.len
		if n == 0 {
			return 0
		}
		a := s.Alignof(t.elem)
		z := s.Sizeof(t.elem)
		return align(z, a)*(n-1) + z
	case *Slice:
		return s.WordSize * 3
	case *Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		offsets := t.offsets
		if t.offsets == nil {
			// compute offsets on demand
			offsets = s.Offsetsof(t.fields)
			t.offsets = offsets
		}
		return offsets[n-1] + s.Sizeof(t.fields[n-1].typ)
	case *Interface:
		return s.WordSize * 2
	}
	return s.WordSize // catch-all
}

// stdSizes is used if Config.Sizes == nil.
var stdSizes = StdSizes{8, 8}

func (conf *Config) alignof(T Type) int64 {
	if s := conf.Sizes; s != nil {
		if a := s.Alignof(T); a >= 1 {
			return a
		}
		panic("Config.Sizes.Alignof returned an alignment < 1")
	}
	return stdSizes.Alignof(T)
}

func (conf *Config) offsetsof(T *Struct) []int64 {
	offsets := T.offsets
	if offsets == nil && T.NumFields() > 0 {
		// compute offsets on demand
		if s := conf.Sizes; s != nil {
			offsets = s.Offsetsof(T.fields)
			// sanity checks
			if len(offsets) != T.NumFields() {
				panic("Config.Sizes.Offsetsof returned the wrong number of offsets")
			}
			for _, o := range offsets {
				if o < 0 {
					panic("Config.Sizes.Offsetsof returned an offset < 0")
				}
			}
		} else {
			offsets = stdSizes.Offsetsof(T.fields)
		}
		T.offsets = offsets
	}
	return offsets
}

// offsetof returns the offset of the field specified via
// the index sequence relative to typ. All embedded fields
// must be structs (rather than pointer to structs).
func (conf *Config) offsetof(typ Type, index []int) int64 {
	var o int64
	for _, i := range index {
		s := typ.Underlying().(*Struct)
		o += conf.offsetsof(s)[i]
		typ = s.fields[i].typ
	}
	return o
}

func (conf *Config) sizeof(T Type) int64 {
	if s := conf.Sizes; s != nil {
		if z := s.Sizeof(T); z >= 0 {
			return z
		}
		panic("Config.Sizes.Sizeof returned a size < 0")
	}
	return stdSizes.Sizeof(T)
}

// align returns the smallest y >= x such that y % a == 0.
func align(x, a int64) int64 {
	y := x + a - 1
	return y - y%a
}
