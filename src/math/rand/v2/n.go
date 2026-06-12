// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: When we drop support for nogenericmethods, merge this into
// rand.go and rewrite the package-level N to "return globalRand.N(n)"

//go:build goexperiment.genericmethods

package rand

// N returns a pseudo-random number in the half-open interval [0,n).
// The type parameter Int can be any integer type.
// It panics if n <= 0.
func (r *Rand) N[Int intType](n Int) Int {
	if n <= 0 {
		panic("invalid argument to N")
	}
	return Int(r.uint64n(uint64(n)))
}
