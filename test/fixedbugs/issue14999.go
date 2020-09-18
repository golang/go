// errorcheck -+

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x int) func(int) int {
	return func(y int) int { return x + y } // ERROR "heap-allocated closure, not allowed in runtime"
}

func g(x int) func(int) int { // ERROR "x escapes to heap, not allowed in runtime"
	return func(y int) int { // ERROR "heap-allocated closure, not allowed in runtime"
		x += y
		return x + y
	}
}
