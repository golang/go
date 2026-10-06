// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"internal/strconv"
	"sync/atomic"
	"unsafe"
)

// Element types

// Floats is all float element types.
type Floats interface {
	float32 | float64
}

// Ints is all signed int element types.
type Ints interface {
	int8 | int16 | int32 | int64
}

// Uints is all unsigned uint element types.
type Uints interface {
	uint8 | uint16 | uint32 | uint64
}

// Nums is all numeric element types.
type Nums interface {
	Floats | Ints | Uints
}

// Mask element types can be used in a Vec to represent a mask.
//
// We use uintN types for these so that regular math for element width and lanes
// work as expected. In a sense, these act like a "wide mask": logically these
// are bool values, represented as either 0 for false or ^0 for true.
type (
	Mask8  uint8
	Mask16 uint16
	Mask32 uint32
	Mask64 uint64
)

// MaskElt is all mask element types. In function signatures, these must always
// be used as the element to a Vec type. They cannot be standalone.
type MaskElt interface {
	Mask8 | Mask16 | Mask32 | Mask64
}

// Elt is all regular (non-mask) vector element types.
type Elt interface {
	Nums
}

// EltOrMask is a constraint that accepts any regular vector element type or
// mask element type.
//
// This type is known to specgen.
type EltOrMask interface {
	Elt | MaskElt
}

// Widths

// Width is a constraint that accepts any type representing a vector width.
//
// This type is known to specgen.
type Width interface {
	Width128 | Width256 | Width512 | WidthScalable
	bits() int // Minimum is 128
}

// FixedWidth is a constraint for all fixed (non-scalable) vector width types.
type FixedWidth interface {
	Width128 | Width256 | Width512
	bits() int
}

type Width128 struct{}

func (Width128) bits() int { return 128 }

type Width256 struct{}

func (Width256) bits() int { return 256 }

type Width512 struct{}

func (Width512) bits() int { return 512 }

// maxScalableWidth is the maximum bit width allowed for scalable vectors.
// This matches the architectural maximum of ARM SVE (2048 bits) and ensures
// that 8-bit unsigned element indices (values 0..255) can address all lanes
// of a byte vector (256 * 8 = 2048 bits) without overflow.
const maxScalableWidth = 2048

// scalableWidth is the active scalable vector bit width. If 0, it's unset and
// [ScalableWidth] panics.
var scalableWidth atomic.Int64

// ScalableWidth returns the current bit width used for scalable vectors when
// executing the spec, or panics if it has not been set via [SetScalableWidth].
func ScalableWidth() int {
	w := int(scalableWidth.Load())
	if w == 0 {
		panic("spec: scalable vector width has not been set (use SetScalableWidth)")
	}
	return w
}

// SetScalableWidth sets the bit width to use for scalable vectors when
// executing the spec (for example, to match the runtime vector length of a
// target hardware architecture during conformance testing).
//
// bits must be a positive multiple of 128 and at most 2048 bits.
//
// SetScalableWidth returns a restore function that clears the scalable width.
//
// Since the scalable width is global state, this panics if there are
// overlapping attempts to set the width.
func SetScalableWidth(bits int) (restore func()) {
	if bits <= 0 || bits > maxScalableWidth || bits%128 != 0 {
		panic("invalid scalable width " + strconv.Itoa(bits))
	}

	if !scalableWidth.CompareAndSwap(0, int64(bits)) {
		panic("SetScalableWidth called concurrently or without restoring previous width")
	}

	return func() {
		if !scalableWidth.CompareAndSwap(int64(bits), 0) {
			panic("restore called out of sequence or more than once")
		}
	}
}

// WidthScalable is the width representing scalable vectors. At a spec level,
// the actual width this represents is completely symbolic, but when executing
// the spec, we concretely interpret this as [ScalableWidth] bits (which must be
// configured via [SetScalableWidth]).
//
// This type is known to specgen.
type WidthScalable struct{}

func (WidthScalable) bits() int { return ScalableWidth() }

// Vectors

// Vec is a vector or mask consisting of the given element type E expanded out
// to W total bits.
//
// This is implemented as a Go slice, which must have length `width[W]() / elemBits[E]()`.
//
// This type is known to specgen.
type Vec[E EltOrMask, W Width] []E

func (v Vec[E, W]) len() int {
	return lanes[E, W]()
}

func makeVec[E EltOrMask, W Width]() Vec[E, W] {
	return make([]E, lanes[E, W]())
}

func elemBits[E EltOrMask]() int {
	return 8 * int(unsafe.Sizeof(*new(E)))
}

func width[W Width]() int {
	return (*(new(W))).bits()
}

func lanes[E EltOrMask, W Width]() int {
	return width[W]() / elemBits[E]()
}

// Other types

// Array represents an array of lanes[E,W]() elements.
//
// The static generator will translate this to a Go array type.
//
// This type is known to specgen.
type Array[E Elt, W Width] []E
