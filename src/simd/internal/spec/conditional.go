// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Conditional logic

package spec

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
//
//specgen:require maskN=xN
func Masked[E Elt, W Width, mE MaskElt](x Vec[E, W], mask Vec[mE, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i, xi := range x {
		if mask[i] != 0 {
			z[i] = xi
		}
	}
	return z
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
//
//specgen:require maskN=xN
func IfElse[E Elt, W Width, mE MaskElt](x Vec[E, W], mask Vec[mE, W], y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i, xi := range x {
		if mask[i] != 0 {
			z[i] = xi
		} else {
			z[i] = y[i]
		}
	}
	return z
}
