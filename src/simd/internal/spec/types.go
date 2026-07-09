// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
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

// scalableWidth is the bit width to use for scalable vectors when executing the
// spec. This is intentionally set to be large and different from any hardware
// platform we support.
//
// TODO: For testing against the spec, do we need a way to match this to the
// hardware we're testing on?
const scalableWidth = 4096

// WidthScalable is the width representing scalable vectors. At a spec level,
// the actual width this represents is completely symbolic, but when executing
// the spec, we concretely interpret this as [scalableWidth] bits.
//
// This type is known to specgen.
type WidthScalable struct{}

func (WidthScalable) bits() int { return scalableWidth }

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
