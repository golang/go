// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

// UintN represents a Go uintN type. An argument x of type UintN introduces a
// constraint variable named xN that must be resolved to the bit width (8, 16,
// 32, or 64). Widths smaller than 8 are rounded up to 8. Widths larger than 64
// are not allowed.
//
// This is used by mask operations that convert between bits in a uintN type and
// elements in a mask.
//
// This type is known to specgen.
type UintN uint64

// MaskFromBits constructs a mask from a bitmap value. If bit i of y is set,
// then mask element i of the result is set.
//
//specgen:name {z}FromBits
//specgen:require x=uint{zL}
func MaskFromBits[E MaskElt, W FixedWidth](x UintN) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		if x&(1<<i) != 0 {
			z[i] = 1
		}
	}
	return z
}

// MaskToBits constructs a bitmap from mask x, where bit i is set if mask
// element i is set.
//
//specgen:name ToBits
//specgen:require z=uint{xL}
func MaskToBits[E MaskElt, W FixedWidth](x Vec[E, W]) (z UintN) {
	for i, elt := range x {
		if elt != 0 {
			z |= 1 << i
		}
	}
	return z
}

// MaskToZ converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true".
//
//specgen:name To{z}
//specgen:require z=Int{xN}x{xL}
func MaskToZ[E MaskElt, W Width, zE Ints](x Vec[E, W]) (z Vec[zE, W]) {
	z = makeVec[zE, W]()
	for i, val := range x {
		if val != 0 {
			z[i] = ^0
		}
	}
	return z
}
