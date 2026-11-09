// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Masks

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

// MaskFromBits returns bitmap x as a mask. If bit i of x is set, then mask
// element i of the result is true.
//
//specgen:name {{.z}}FromBits
//specgen:require x=uint{zL}
func MaskFromBits[E MaskElt, W FixedWidth](x UintN) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		if x&(1<<i) != 0 {
			z[i] = ^E(0)
		}
	}
	return z
}

// MaskToBits returns x as a bitmap, where bit i is set if mask element i is
// true.
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

// MaskAllTrue returns a mask where every element is true.
//
//specgen:name {{.z}}AllTrue
//specgen:experiment
func MaskAllTrue[E MaskElt, W Width]() (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = ^E(0)
	}
	return z
}

// MaskToZ converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true", and 0 otherwise.
//
//specgen:name To{{.z}}
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

// All returns true when all positions in mask x are true.
func All[E MaskElt, W Width](x Vec[E, W]) bool {
	for _, elt := range x {
		if elt == 0 {
			return false
		}
	}
	return true
}

// Any returns true when any position in mask x is true.
func Any[E MaskElt, W Width](x Vec[E, W]) bool {
	for _, elt := range x {
		if elt != 0 {
			return true
		}
	}
	return false
}

// None returns true when no positions in mask x are set.
func None[E MaskElt, W Width](x Vec[E, W]) bool {
	return !Any(x)
}

// MaskFromCount returns a mask whose first count elements are true and the rest
// false. A count <= 0 gives no true elements, and a count >= {{lanes .z "z"}}
// gives all true elements.
//
//	z[i] = i < count
//
//specgen:name {{.z}}FromCount
//specgen:experiment
func MaskFromCount[E MaskElt, W Width](count int) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		if i < count {
			z[i] = ^E(0)
		}
	}
	return z
}

// First returns a one-hot mask with only the lowest true element of x, or an
// all-false mask if x is all false.
//
//	for i, elt := range x {
//		if elt {
//			z[i] = true
//			break
//		}
//	}
//
//specgen:experiment
func First[E MaskElt, W Width](x Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i, elt := range x {
		if elt != 0 {
			z[i] = ^E(0)
			break
		}
	}
	return z
}
// Next returns a one-hot mask with the lowest true element of x that is above
// the highest true element of cursor, or an all-false mask if there is no such
// element. If cursor has no true elements, it returns a one-hot mask with the
// first true element of x.
//
//	start := 0
//	for i, c := range cursor {
//		if c {
//			start = i + 1
//		}
//	}
//	for i := start; i < x.Len(); i++ {
//		if x[i] {
//			z[i] = true
//			break
//		}
//	}
//
// Together with [First], this steps through the true elements of a mask one at
// a time:
//
//	for cursor := x.First(); cursor.Any(); cursor = x.Next(cursor) {
//		...
//	}
//
//specgen:experiment
func Next[E MaskElt, W Width](x, cursor Vec[E, W]) (z Vec[E, W]) {
	// TODO: This is pretty SVE-specific. How does this compare with RVV?
	start := 0
	for i, c := range cursor {
		if c != 0 {
			start = i + 1
		}
	}
	z = makeVec[E, W]()
	for i := start; i < x.len(); i++ {
		if x[i] != 0 {
			z[i] = ^E(0)
			break
		}
	}
	return z
}
