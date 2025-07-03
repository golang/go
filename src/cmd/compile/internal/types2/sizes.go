// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Sizes.

package types2

// Sizes defines the sizing functions for package unsafe.
type Sizes interface {
	// Alignof returns the alignment of a variable of type T.
	// Alignof must implement the alignment guarantees required by the spec.
	// The result must be >= 1.
	Alignof(T Type) int64

	// Offsetsof returns the offsets of the given struct fields, in bytes.
	// Offsetsof must implement the offset guarantees required by the spec.
	// A negative entry in the result indicates that the struct is too large.
	Offsetsof(fields []*Var) []int64

	// Sizeof returns the size of a variable of type T.
	// Sizeof must implement the size guarantees required by the spec.
	// A negative result indicates that T is too large.
	Sizeof(T Type) int64
}

// StdSizes is a convenience type for creating commonly used Sizes.
// It makes the following simplifying assumptions:
//
//   - The size of explicitly sized basic types (int16, etc.) is the
//     specified size.
//   - The size of strings and interfaces is 2*WordSize.
//   - The size of slices is 3*WordSize.
//   - The size of an array of n elements corresponds to the size of
//     a struct of n consecutive fields of the array's element type.
//   - The size of a struct is the offset of the last field plus that
//     field's size. As with all element types, if the struct is used
//     in an array its size must first be aligned to a multiple of the
//     struct's alignment.
//   - All other types have size WordSize.
//   - Arrays and structs are aligned per spec definition; all other
//     types are naturally aligned with a maximum alignment MaxAlign.
//
// *StdSizes implements Sizes.
type StdSizes struct {
	WordSize int64 // word size in bytes - must be >= 4 (32bits)
	MaxAlign int64 // maximum alignment in bytes - must be >= 1
}

func (s *StdSizes) Alignof(T Type) (result int64) {
	defer func() {
		assert(result >= 1)
	}()

	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := under(T).(type) {
	case *Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return s.Alignof(t.elem)
	case *Struct:
		if len(t.fields) == 0 && IsSyncAtomicAlign64(T) {
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
		for _, f := range t.fields {
			if a := s.Alignof(f.typ); a > max {
				max = a
			}
		}
		return max
	case *Slice, *Interface:
		// Multiword data structures are effectively structs
		// in which each element has size WordSize.
		// Type parameters lead to variable sizes/alignments;
		// StdSizes.Alignof won't be called for them.
		assert(!isTypeParam(T))
		return s.WordSize
	case *Basic:
		// Strings are like slices and interfaces.
		if t.Info()&IsString != 0 {
			return s.WordSize
		}
	case *TypeParam, *Union:
		panic("unreachable")
	}
	a := s.Sizeof(T) // may be 0 or negative
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

func IsSyncAtomicAlign64(T Type) bool {
	named := asNamed(T)
	if named == nil {
		return false
	}
	obj := named.Obj()
	return obj.Name() == "align64" &&
		obj.Pkg() != nil &&
		(obj.Pkg().Path() == "sync/atomic" ||
			obj.Pkg().Path() == "internal/runtime/atomic")
}

func (s *StdSizes) Offsetsof(fields []*Var) []int64 {
	offsets := make([]int64, len(fields))
	var offs int64
	for i, f := range fields {
		if offs < 0 {
			// all remaining offsets are too large
			offsets[i] = -1
			continue
		}
		// offs >= 0
		a := s.Alignof(f.typ)
		offs = align(offs, a) // possibly < 0 if align overflows
		offsets[i] = offs
		if d := s.Sizeof(f.typ); d >= 0 && offs >= 0 {
			offs += d // ok to overflow to < 0
		} else {
			offs = -1 // f.typ or offs is too large
		}
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
	switch t := under(T).(type) {
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
		if n <= 0 {
			return 0
		}
		// n > 0
		esize := s.Sizeof(t.elem)
		if esize < 0 {
			return -1 // element too large
		}
		if esize == 0 {
			return 0 // 0-size element
		}
		// esize > 0
		a := s.Alignof(t.elem)
		ea := align(esize, a) // possibly < 0 if align overflows
		if ea < 0 {
			return -1
		}
		// ea >= 1
		n1 := n - 1 // n1 >= 0
		// Final size is ea*n1 + esize; and size must be <= maxInt64.
		const maxInt64 = 1<<63 - 1
		if n1 > 0 && ea > maxInt64/n1 {
			return -1 // ea*n1 overflows
		}
		return ea*n1 + esize // may still overflow to < 0 which is ok
	case *Slice:
		return s.WordSize * 3
	case *Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		offsets := s.Offsetsof(t.fields)
		offs := offsets[n-1]
		size := s.Sizeof(t.fields[n-1].typ)
		if offs < 0 || size < 0 {
			return -1 // type too large
		}
		return offs + size // may overflow to < 0 which is ok
	case *Interface:
		// Type parameters lead to variable sizes/alignments;
		// StdSizes.Sizeof won't be called for them.
		assert(!isTypeParam(T))
		return s.WordSize * 2
	case *TypeParam, *Union:
		panic("unreachable")
	}
	return s.WordSize // catch-all
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
	// update the doc string of SizesFor below.
}

// SizesFor returns the Sizes used by a compiler for an architecture.
// The result is nil if a compiler/architecture pair is not known.
//
// Supported architectures for compiler "gc":
// "386", "amd64", "amd64p32", "arm", "arm64", "loong64", "mips", "mipsle",
// "mips64", "mips64le", "ppc64", "ppc64le", "riscv64", "s390x", "sparc64", "wasm".
func SizesFor(compiler, arch string) Sizes {
	switch compiler {
	case "gc":
		if s := gcSizesFor(compiler, arch); s != nil {
			return Sizes(s)
		}
	case "gccgo":
		if s, ok := gccgoArchSizes[arch]; ok {
			return Sizes(s)
		}
	}
	return nil
}

// stdSizes is used if Config.Sizes == nil.
var stdSizes = SizesFor("gc", "amd64")

func (conf *Config) alignof(T Type) int64 {
	f := stdSizes.Alignof
	if conf.Sizes != nil {
		f = conf.Sizes.Alignof
	}
	if a := f(T); a >= 1 {
		return a
	}
	panic("implementation of alignof returned an alignment < 1")
}

func (conf *Config) offsetsof(T *Struct) []int64 {
	var offsets []int64
	if T.NumFields() > 0 {
		// compute offsets on demand
		f := stdSizes.Offsetsof
		if conf.Sizes != nil {
			f = conf.Sizes.Offsetsof
		}
		offsets = f(T.fields)
		// sanity checks
		if len(offsets) != T.NumFields() {
			panic("implementation of offsetsof returned the wrong number of offsets")
		}
	}
	return offsets
}

// offsetof returns the offset of the field specified via
// the index sequence relative to T. All embedded fields
// must be structs (rather than pointers to structs).
// If the offset is too large (because T is too large),
// the result is negative.
func (conf *Config) offsetof(T Type, index []int) int64 {
	var offs int64
	for _, i := range index {
		s := under(T).(*Struct)
		d := conf.offsetsof(s)[i]
		if d < 0 {
			return -1
		}
		offs += d
		if offs < 0 {
			return -1
		}
		T = s.fields[i].typ
	}
	return offs
}

// sizeof returns the size of T.
// If T is too large, the result is negative.
func (conf *Config) sizeof(T Type) int64 {
	f := stdSizes.Sizeof
	if conf.Sizes != nil {
		f = conf.Sizes.Sizeof
	}
	return f(T)
}

// align returns the smallest y >= x such that y % a == 0.
// a must be within 1 and 8 and it must be a power of 2.
// The result may be negative due to overflow.
func align(x, a int64) int64 {
	assert(x >= 0 && 1 <= a && a <= 8 && a&(a-1) == 0)
	return (x + a - 1) &^ (a - 1)
}
