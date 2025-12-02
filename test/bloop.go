// errorcheck -0 -m=2

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test keeping statements results in testing.B.Loop alive.
// See issue #61515, #73137.

package foo

import "testing"

func caninline(x int) int { // ERROR "can inline caninline"
	return x
}

var something int

func caninlineNoRet(x int) { // ERROR "can inline caninlineNoRet"
	something = x
}

func caninlineVariadic(x ...int) { // ERROR "can inline caninlineVariadic" "x does not escape"
	something = x[0]
}

func test(b *testing.B, localsink, cond int) { // ERROR ".*"
	for i := 0; i < b.N; i++ {
		caninline(1) // ERROR "inlining call to caninline"
	}
	somethingptr := &something
	for b.Loop() { // ERROR "inlining call to testing\.\(\*B\)\.Loop"
		caninline(1)                 // ERROR "inlining call to caninline" "function result will be kept alive"
		caninlineNoRet(1)            // ERROR "inlining call to caninlineNoRet" "function arg will be kept alive"
		caninlineVariadic(1)         // ERROR "inlining call to caninlineVariadic" "function arg will be kept alive" ".* does not escape"
		caninlineVariadic(localsink) // ERROR "inlining call to caninlineVariadic" "localsink will be kept alive" ".* does not escape"
		localsink = caninline(1)     // ERROR "inlining call to caninline" "localsink will be kept alive"
		localsink += 5               // ERROR "localsink will be kept alive"
		localsink, cond = 1, 2       // ERROR "localsink will be kept alive" "cond will be kept alive"
		*somethingptr = 1            // ERROR "dereference will be kept alive"
		if cond > 0 {
			caninline(1) // ERROR "inlining call to caninline" "function result will be kept alive"
		}
		switch cond {
		case 2:
			caninline(1) // ERROR "inlining call to caninline" "function result will be kept alive"
		}
		{
			caninline(1) // ERROR "inlining call to caninline" "function result will be kept alive"
		}
	}
}
