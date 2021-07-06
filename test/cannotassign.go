// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test "cannot assign" errors

package main

func main() {
	var s string = "hello"
	s[1:2] = "a" // ERROR "cannot assign to .* (\(strings are immutable\))?"
	s[3] = "b"   // ERROR "cannot assign to .* (\(strings are immutable\))?"

	const n int = 1
	const cs string = "hello"
	n = 2        // ERROR "cannot assign to .* (\(declared const\))?"
	cs = "hi"    // ERROR "cannot assign to .* (\(declared const\))?"
	true = false // ERROR "cannot assign to .* (\(declared const\))?"

	var m map[int]struct{ n int }
	m[0].n = 7 // ERROR "cannot assign to struct field .* in map$"

	1 = 7         // ERROR "cannot assign to 1"
	"hi" = 7      // ERROR `cannot assign to "hi"`
	nil = 7       // ERROR "cannot assign to nil"
	len("") = 7   // ERROR `cannot assign to len\(""\)`
	[]int{} = nil // ERROR "cannot assign to \[\]int\{\}"

	var x int = 7
	x + 1 = 7 // ERROR "cannot assign to x \+ 1"
}
