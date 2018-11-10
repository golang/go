// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that erroneous switch statements are detected by the compiler.
// Does not compile.

package main

func f() {
	switch {
	case 0; // ERROR "expecting := or = or : or comma|expecting :"
	}

	switch {
	case 0; // ERROR "expecting := or = or : or comma|expecting :"
	default:
	}

	switch {
	case 0: case 0: default:
	}

	switch {
	case 0: f(); case 0:
	case 0: f() case 0: // ERROR "unexpected case at end of statement"
	}

	switch {
	case 0: f(); default:
	case 0: f() default: // ERROR "unexpected default at end of statement"
	}

	switch {
	if x: // ERROR "expecting case or default or }"
	}
}
