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

// InterleaveLo interleaves the elements of the low halves of x and y.
//
// {{if eq .xL 2}}
//
//	z = {x[0], y[0]}
//
// {{else if eq .xL 4}}
//
//	z = {x[0], y[0], x[1], y[1]}
//
// {{else}}
//
//	z = {x[0], y[0], x[1], y[1], ..., x[n/2-1], y[n/2-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:experiment
func InterleaveLo[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z.len() / 2 {
		z[2*i] = x[i]
		z[2*i+1] = y[i]
	}
	return z
}

// InterleaveHi interleaves the elements of the high halves of x and y.
//
// {{if eq .xL 2}}
//
//	z = {x[1], y[1]}
//
// {{else if eq .xL 4}}
//
//	z = {x[2], y[2], x[3], y[3]}
//
// {{else}}
//
//	z = {x[n/2], y[n/2], x[n/2+1], y[n/2+1], ..., x[n-1], y[n-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:experiment
func InterleaveHi[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := range half {
		z[2*i] = x[half+i]
		z[2*i+1] = y[half+i]
	}
	return z
}

// ConcatEven concatenates the even-indexed elements of x and y.
//
// {{if eq .xL 2}}
//
//	z = {x[0], y[0]}
//
// {{else if eq .xL 4}}
//
//	z = {x[0], x[2], y[0], y[2]}
//
// {{else}}
//
//	z = {x[0], x[2], ..., x[n-2], y[0], y[2], ..., y[n-2]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:experiment
func ConcatEven[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := range half {
		z[i] = x[2*i]
		z[half+i] = y[2*i]
	}
	return z
}

// ConcatOdd concatenates the odd-indexed elements of x and y.
//
// {{if eq .xL 2}}
//
//	z = {x[1], y[1]}
//
// {{else if eq .xL 4}}
//
//	z = {x[1], x[3], y[1], y[3]}
//
// {{else}}
//
//	z = {x[1], x[3], ..., x[n-1], y[1], y[3], ..., y[n-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:experiment
func ConcatOdd[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := range half {
		z[i] = x[2*i+1]
		z[half+i] = y[2*i+1]
	}
	return z
}
