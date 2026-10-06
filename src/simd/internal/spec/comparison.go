// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Comparison

package spec

import "math"

func bool2mask[E MaskElt](b bool) E {
	if b {
		return ^E(0)
	}
	return 0
}

// Equal returns a mask indicating which elements of x and y are equal.
//
//specgen:require zN=xN
func Equal[E EltOrMask, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x == y)
	})
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
//
//specgen:require zN=xN
func NotEqual[E EltOrMask, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x != y)
	})
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
//
//specgen:require zN=xN
func Less[E Elt, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x < y)
	})
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
//
//specgen:require zN=xN
func LessEqual[E Elt, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x <= y)
	})
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
//
//specgen:require zN=xN
func Greater[E Elt, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x > y)
	})
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
//
//specgen:require zN=xN
func GreaterEqual[E Elt, W Width, zE MaskElt](x, y Vec[E, W]) (z Vec[zE, W]) {
	return map2[E, W, zE, W](x, y, func(x, y E) zE {
		return bool2mask[zE](x >= y)
	})
}

// ToMask returns a mask indicating which elements of x are non-zero.
//
//	z[i] = x[i] != 0
//
//specgen:require zN=xN
func ToMask[E Elt, W Width, zE MaskElt](x Vec[E, W]) (z Vec[zE, W]) {
	return map1[E, W, zE, W](x, func(x E) zE {
		return bool2mask[zE](x != 0)
	})
}

// IsNan returns a mask indicating which elements of x are NaN.
//
//	z[i] = math.IsNaN(x[i])
//
//specgen:require zN=xN
func IsNaN[E Floats, W Width, zE MaskElt](x Vec[E, W]) (z Vec[zE, W]) {
	return map1[E, W, zE, W](x, func(x E) zE {
		return bool2mask[zE](math.IsNaN(float64(x)))
	})
}
