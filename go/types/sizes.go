// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support for (unsafe) Alignof, Offsetof, and Sizeof.

package types

func (conf *Config) alignof(typ Type) int64 {
	if f := conf.Alignof; f != nil {
		if a := f(typ); a >= 1 {
			return a
		}
		panic("Config.Alignof returned an alignment < 1")
	}
	return DefaultAlignof(typ)
}

func (conf *Config) offsetsof(s *Struct) []int64 {
	offsets := s.offsets
	if offsets == nil && s.NumFields() > 0 {
		// compute offsets on demand
		if f := conf.Offsetsof; f != nil {
			offsets = f(s.fields)
			// sanity checks
			if len(offsets) != s.NumFields() {
				panic("Config.Offsetsof returned the wrong number of offsets")
			}
			for _, o := range offsets {
				if o < 0 {
					panic("Config.Offsetsof returned an offset < 0")
				}
			}
		} else {
			offsets = DefaultOffsetsof(s.fields)
		}
		s.offsets = offsets
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

func (conf *Config) sizeof(typ Type) int64 {
	if f := conf.Sizeof; f != nil {
		if s := f(typ); s >= 0 {
			return s
		}
		panic("Config.Sizeof returned a size < 0")
	}
	return DefaultSizeof(typ)
}

// DefaultMaxAlign is the default maximum alignment, in bytes,
// used by DefaultAlignof.
const DefaultMaxAlign = 8

// DefaultAlignof implements the default alignment computation
// for unsafe.Alignof. It is used if Config.Alignof == nil.
func DefaultAlignof(typ Type) int64 {
	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := typ.Underlying().(type) {
	case *Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return DefaultAlignof(t.elt)
	case *Struct:
		// spec: "For a variable x of struct type: unsafe.Alignof(x)
		// is the largest of the values unsafe.Alignof(x.f) for each
		// field f of x, but at least 1."
		max := int64(1)
		for _, f := range t.fields {
			if a := DefaultAlignof(f.typ); a > max {
				max = a
			}
		}
		return max
	}
	a := DefaultSizeof(typ) // may be 0
	// spec: "For a variable x of any type: unsafe.Alignof(x) is at least 1."
	if a < 1 {
		return 1
	}
	if a > DefaultMaxAlign {
		return DefaultMaxAlign
	}
	return a
}

// align returns the smallest y >= x such that y % a == 0.
func align(x, a int64) int64 {
	y := x + a - 1
	return y - y%a
}

// DefaultOffsetsof implements the default field offset computation
// for unsafe.Offsetof. It is used if Config.Offsetsof == nil.
func DefaultOffsetsof(fields []*Var) []int64 {
	offsets := make([]int64, len(fields))
	var o int64
	for i, f := range fields {
		a := DefaultAlignof(f.typ)
		o = align(o, a)
		offsets[i] = o
		o += DefaultSizeof(f.typ)
	}
	return offsets
}

// DefaultPtrSize is the default size of ints, uint, and pointers, in bytes,
// used by DefaultSizeof.
const DefaultPtrSize = 8

// DefaultSizeof implements the default size computation
// for unsafe.Sizeof. It is used if Config.Sizeof == nil.
func DefaultSizeof(typ Type) int64 {
	switch t := typ.Underlying().(type) {
	case *Basic:
		if s := t.size; s > 0 {
			return s
		}
		if t.kind == String {
			return DefaultPtrSize * 2
		}
	case *Array:
		a := DefaultAlignof(t.elt)
		s := DefaultSizeof(t.elt)
		return align(s, a) * t.len // may be 0
	case *Slice:
		return DefaultPtrSize * 3
	case *Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		offsets := t.offsets
		if t.offsets == nil {
			// compute offsets on demand
			offsets = DefaultOffsetsof(t.fields)
			t.offsets = offsets
		}
		return offsets[n-1] + DefaultSizeof(t.fields[n-1].typ)
	case *Signature:
		return DefaultPtrSize * 2
	}
	return DefaultPtrSize // catch-all
}
