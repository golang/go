// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Shuffles

package spec

// Permute permutes x.
//
//	z[i] = x[indices[i] % len(x)]
//
//specgen:require indices=Uint{xN}x{xL}
func Permute[E Elt, W Width, E2 Uints](x Vec[E, W], indices Vec[E2, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = x[uint(indices[i])%uint(x.len())]
	}
	return z
}

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if 0 <= indices[i] && indices[i] < {{lanes .x "x"}} {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
func LookupOrZero[E Ints | Uints, W Width](x, indices Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i, idx := range indices {
		if idx >= 0 && uint64(idx) < uint64(x.len()) {
			z[i] = x[idx]
		}
	}
	return z
}
