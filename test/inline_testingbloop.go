// errorcheck -0 -m=2

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test no inlining of function calls in testing.B.Loop.
// See issue #61515.

package foo

import "testing"

func caninline(x int) int { // ERROR "can inline caninline"
	return x
}

func cannotinline(b *testing.B) { // ERROR "b does not escape" "cannot inline cannotinline.*"
	for i := 0; i < b.N; i++ {
		caninline(1) // ERROR "inlining call to caninline"
	}
	for b.Loop() { // ERROR "skip inlining within testing.B.loop" "inlining call to testing\.\(\*B\)\.Loop"
		caninline(1)
	}
	for i := 0; i < b.N; i++ {
		caninline(1) // ERROR "inlining call to caninline"
	}
	for b.Loop() { // ERROR "skip inlining within testing.B.loop" "inlining call to testing\.\(\*B\)\.Loop"
		caninline(1)
	}
	for i := 0; i < b.N; i++ {
		caninline(1) // ERROR "inlining call to caninline"
	}
	for b.Loop() { // ERROR "skip inlining within testing.B.loop" "inlining call to testing\.\(\*B\)\.Loop"
		caninline(1)
	}
}
