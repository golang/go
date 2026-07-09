// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

// Add adds corresponding elements of two vectors.
//
//	z[i] = x[i] + y[i]
func Add[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E { return x + y })
}

// DotProductPairs multiplies corresponding elements of x and y, and sums
// adjacent pairs, yielding a vector of half as many elements with twice the
// input element size.
//
//	w[i] = x[i] * y[i]        // Double width
//	z[i] = w[2*i] + w[2*i+1]
//
//specgen:require z={xB}{xN*2}x{xL/2}
func DotProductPairs[E Nums, W Width, zE Nums](x, y Vec[E, W]) (z Vec[zE, W]) {
	// TODO: How do we handle/specify overflow? x86 only supports this on signed
	// types, and the only case that can overflow is if all four elements are
	// MinInt16 (in which case the true result is MaxInt32+1, which wraps around
	// to MinInt32). Unsigned types can overflow much more readily.
	//
	// Maybe we just leave overflow unspecified (or "architecture dependent").
	// In which case, we probably need a way to communicate that in the spec
	// (designated panic?).
	//
	// We might also need a way to constraint this to same-signed E and zE,
	// which the constraint language doesn't currently have a way to say, but we
	// could add as a built-in projection function in the syntax.
	z = makeVec[zE, W]()
	for i := range z {
		z[i] = zE(x[2*i])*zE(y[2*i]) + zE(x[2*i+1])*zE(y[2*i+1])
	}
	return z
}

// DotProductPairsSaturated multiplies corresponding elements of x and y, and
// sums adjacent pairs, all with saturation. It yields a vector of half as many
// elements with twice the input element size.
//
//	w[i] = x[i] * y[i]        // Double width, saturated
//	z[i] = w[2*i] + w[2*i+1]  // Saturated
//
//specgen:require y=Int{xN}x{xL} z=Int{xN*2}x{xL/2}
func DotProductPairsSaturated[xE Uints, xW Width, yE Ints, zE Ints](x Vec[xE, xW], y Vec[yE, xW]) (z Vec[zE, xW]) {
	z = makeVec[zE, xW]()
	for i := range z {
		a := mulSaturatedUSS64(uint64(x[2*i]), int64(y[2*i]))
		b := mulSaturatedUSS64(uint64(x[2*i+1]), int64(y[2*i+1]))
		z[i] = saturateS[zE](addSaturatedSSS64(a, b))
	}
	return z
}
