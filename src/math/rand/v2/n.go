//go:build goexperiment.genericmethods

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package rand

// N returns a pseudo-random number in the half-open interval [0,n).
// The type parameter Int can be any integer type.
// It panics if n <= 0.
func (r *Rand) N[Int intType](n Int) Int {
	// TODO: See CL 775100,
	// When delete the goexperiment.genericmethods,
	// make this method
	// share the implementation with the global function N
	// and delete this file.
	if n <= 0 {
		panic("invalid argument to N")
	}
	return Int(r.uint64n(uint64(n)))
}
