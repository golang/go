// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Reductions

package spec

// ReduceSumInts returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
//specgen:name ReduceSum
func ReduceSumInts[E Ints | Uints, W Width](x Vec[E, W]) (z E) {
	// Integer addition wraps and is associative, so the order doesn't matter.
	for _, xi := range x {
		z += xi
	}
	return z
}

// ReduceSumFloats returns the scalar sum of the elements of x.
//
// {{if eq .xL 2}}
//
//	z = x[0] + x[1]
//
// {{else}}
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
// {{if eq .xL 4}}
//
//	z = (x[0] + x[1]) + (x[2] + x[3])
//
// {{else}}
//
//	z = ((x[0] + x[1]) + (x[2] + x[3])) + ...
//
// {{end}}
//
// {{end}}
//
//specgen:name ReduceSum
func ReduceSumFloats[E Floats, W Width](x Vec[E, W]) (z E) {
	// Vector lengths are powers of two, so every level pairs up evenly.
	tmp := make([]E, x.len()/2, maxScalableWidth/elemBits[E]()/2)
	for i := range tmp {
		tmp[i] = x[2*i] + x[2*i+1]
	}
	for len(tmp) > 1 {
		half := len(tmp) / 2
		for i := range half {
			tmp[i] = tmp[2*i] + tmp[2*i+1]
		}
		tmp = tmp[:half]
	}
	return tmp[0]
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// {{if eq .xB "float"}}
// The handling of NaN and signed zeroes is platform-dependent. See [Max].
// {{end}}
func ReduceMax[E Elt, W Width](x Vec[E, W]) (z E) {
	// TODO: Do we need per-platform handling of non-finites? Since AMD64 is the
	// only platform that's weird about non-finite values and it doesn't have
	// ReduceMax, maybe we don't have to, but if we ever want to emulate this on
	// AMD64, we'd have to decide whether to match AMD64's Max or Go/IEEE Max.
	z = x[0]
	for _, xi := range x[1:] {
		z = max(z, xi)
	}
	return z
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// {{if eq .xB "float"}}
// The handling of NaN and signed zeroes is platform-dependent. See [Min].
// {{end}}
func ReduceMin[E Elt, W Width](x Vec[E, W]) (z E) {
	// See ReduceMax about the handling of non-finite values.
	z = x[0]
	for _, xi := range x[1:] {
		z = min(z, xi)
	}
	return z
}
