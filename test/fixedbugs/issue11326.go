// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for golang.org/issue/11326.

package main

func main() {
	// The gc compiler implementation uses the minimally required 32bit
	// binary exponent, so these constants cannot be represented anymore
	// internally. However, the language spec does not preclude other
	// implementations from handling these. Don't check the error.
	// var _ = 1e2147483647 // "constant too large"
	// var _ = 1e646456993  // "constant too large"

	// Any implementation must be able to handle these constants at
	// compile time (even though they cannot be assigned to a float64).
	var _ = 1e646456992  // ERROR "1e\+646456992 overflows float64"
	var _ = 1e64645699   // ERROR "1e\+64645699 overflows float64"
	var _ = 1e6464569    // ERROR "1e\+6464569 overflows float64"
	var _ = 1e646456     // ERROR "1e\+646456 overflows float64"
	var _ = 1e64645      // ERROR "1e\+64645 overflows float64"
	var _ = 1e6464       // ERROR "1e\+6464 overflows float64"
	var _ = 1e646        // ERROR "1e\+646 overflows float64"
	var _ = 1e309        // ERROR "1e\+309 overflows float64"

	var _ = 1e308
}
