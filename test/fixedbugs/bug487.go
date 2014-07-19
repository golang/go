// errorcheck

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler did not reliably report mismatches between the
// number of function results and the number of expected results.

package p

func G() (int, int, int) {
	return 0, 0, 0
}

func F() {
	a, b := G()	// ERROR "mismatch"
	a, b = G()	// ERROR "mismatch"
	_, _ = a, b
}

func H() (int, int) {
	return G()	// ERROR "too many|mismatch"
}
