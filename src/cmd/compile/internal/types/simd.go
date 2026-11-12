// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"internal/buildcfg"
	"strings"

	"cmd/compile/internal/base"
)

// IsSIMD reports whether t is a SIMD type: one of the SSA vector and mask
// types, such as TypeVec128, or one of the vector and mask types of
// simd/archsimd, or a type defined in terms of one of those, such as
// bridge.Int32x4 in package simd/internal/bridge, which is defined as
// archsimd.Int32x4. Values of SIMD types live in vector registers. The tag
// types that mark the vector and mask types of simd/archsimd are not SIMD
// types.
func (t *Type) IsSIMD() bool {
	return t.flags&typeIsSIMD != 0
}

// ContainsSIMD reports whether t is a SIMD type, or an array or struct
// that contains one.
func (t *Type) ContainsSIMD() bool {
	if t.IsSIMD() {
		return true
	}
	switch t.Kind() {
	case TARRAY:
		return t.Elem().ContainsSIMD()
	case TSTRUCT:
		for _, f := range t.Fields() {
			if f.Type.ContainsSIMD() {
				return true
			}
		}
	}
	return false
}

// SIMDShape describes the elements of a SIMD vector or mask type.
type SIMDShape struct {
	// Elem is the element type. For example, it's float32 for Float32x4.
	//
	// The Elem of a mask is the signed integer type of the size of the
	// vector elements it governs: int32 for Mask32x4 and for Mask32s.
	Elem *Type

	// Lanes is the number of elements of a fixed-width type, or 0 for a
	// scalable type.
	Lanes int64

	// Scalable reports whether this is an arm64 SVE type, whose number of
	// elements is only known at run time.
	Scalable bool

	// Mask reports whether this is a mask type, such as Mask32x4 or
	// Mask32s.
	//
	// The fixed-width masks are stored as vectors whose elements are 0 or
	// -1, so most of the compiler can treat them like vectors. But a
	// scalable mask on arm64 is an SVE predicate: it's stored as a bitmap
	// with one bit per byte of a scalable vector rather than as an array
	// of Elem, and lives in a predicate register.
	Mask bool
}

// SIMDShape returns the shape of SIMD vector or mask type t, and whether t is
// such a type. It works on the vector and mask types of simd/archsimd, and
// types defined in terms of them, such as the types of package simd on
// platforms where it isn't emulated. Each is a struct whose first field is a
// SIMD tag and whose second field holds its elements.
func (t *Type) SIMDShape() (shape SIMDShape, ok bool) {
	if !t.IsSIMD() || !t.IsStruct() {
		// Not a SIMD vector or mask type declared in Go. This also
		// excludes the SSA SIMD types, which aren't structs.
		return SIMDShape{}, false
	}
	if t.NumFields() != 2 || t.Field(0).Type.flags&typeIsSIMDTag == 0 {
		base.Fatalf("SIMD type %v isn't a tag followed by its elements", t)
	}
	tag, vals := t.Field(0), t.Field(1).Type
	// The tag field of a mask type is named after the mask, like mask32x4
	// or mask32s, and that of a vector type after the vector, like
	// float32x4. The rest of their fields can be the same.
	shape.Mask = strings.HasPrefix(tag.Sym.Name, "mask")

	switch tag.Type.Sym().Name {
	case "psve":
		// SVE predicate mask. Its second field is just storage for its
		// bits, so the name of its tag field records the size of the
		// vector elements it governs.
		shape.Scalable = true
		switch tag.Sym.Name {
		case "mask8s":
			shape.Elem = Types[TINT8]
		case "mask16s":
			shape.Elem = Types[TINT16]
		case "mask32s":
			shape.Elem = Types[TINT32]
		case "mask64s":
			shape.Elem = Types[TINT64]
		default:
			base.Fatalf("SVE predicate %v has unexpected tag field %v", t, tag.Sym)
		}
		return shape, true
	case "v256":
		// On arm64, v256 is the tag of the SVE scalable vectors.
		shape.Scalable = buildcfg.GOARCH == "arm64"
	}
	if !vals.IsArray() {
		base.Fatalf("SIMD type %v has elements of non-array type %v", t, vals)
	}
	shape.Elem = vals.Elem()
	if !shape.Scalable {
		shape.Lanes = vals.NumElem()
	}
	return shape, true
}
