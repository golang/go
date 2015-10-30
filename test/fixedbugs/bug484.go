// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The liveness code used to say that, in func g, s was live
// starting at its declaration, because it appears to have its
// address taken by the closure (different s, but the parser
// gets slightly confused, a separate bug). The liveness analysis
// saw s as having its address taken but the register optimizer
// did not. This mismatch meant that s would be marked live
// (and therefore initialized) at the call to f, but the register optimizer
// would optimize away the initialization of s before f, causing the
// garbage collector to use unused data.
// The register optimizer has been changed to respect the
// same "address taken" flag that the liveness analysis uses,
// even if it cannot see any address being taken in the actual
// machine code. This is conservative but keeps the two consistent,
// which is the most important thing.

package main

import "runtime"

//go:noinline
func f() interface{} {
	runtime.GC()
	return nil
}

//go:noinline
func g() {
	var s interface{}
	_ = func() {
		s := f()
		_ = s
	}
	s = f()
	useiface(s)
	useiface(s)
}

//go:noinline
func useiface(x interface{}) {
}

//go:noinline
func h() {
	var x [16]uintptr
	for i := range x {
		x[i] = 1
	}

	useint(x[0])
	useint(x[1])
	useint(x[2])
	useint(x[3])
}

//go:noinline
func useint(x uintptr) {
}

func main() {
	// scribble non-zero values on stack
	h()
	// call function that used to let the garbage collector
	// see uninitialized stack values; it will see the
	// nonzero values.
	g()
}

func big(x int) {
	if x >= 0 {
		big(x - 1)
	}
}
