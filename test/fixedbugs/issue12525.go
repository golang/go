// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12525: confusing error trying to increment boolean value

package main

func main() {
	var i int
	i++

	var f float64
	f++

	var c complex128
	c++

	var b bool
	b++ // ERROR "invalid operation: b\+\+ \(non-numeric type bool\)"

	var s string
	s-- // ERROR "invalid operation: s-- \(non-numeric type string\)"
}
