// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that > 10 non-syntax errors on the same line
// don't lead to early exit. Specifically, here test
// that we see the initialization error for variable
// s.

package main

type T struct{}

func main() {
	t := T{X: 1, X: 1, X: 1, X: 1, X: 1, X: 1, X: 1, X: 1, X: 1, X: 1} // ERROR "unknown field 'X' in struct literal of type T"
	var s string = 1 // ERROR "cannot use 1"
}
