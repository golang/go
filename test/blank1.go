// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that incorrect uses of the blank identifier are caught.
// Does not compile.

package _	// ERROR "invalid package name"

var t struct {
	_ int
}

func (x int) _() { // ERROR "methods on non-local type"
	println(x)
}

type T struct {
      _ []int
}

func main() {
	_()	// ERROR "cannot use .* as value"
	x := _+1	// ERROR "cannot use .* as value"
	_ = x
	_ = t._ // ERROR "cannot refer to blank field|invalid use of|t._ undefined"

      var v1, v2 T
      _ = v1 == v2 // ERROR "cannot be compared|non-comparable|cannot compare v1 == v2"
}
