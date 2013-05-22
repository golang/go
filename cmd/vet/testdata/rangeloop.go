// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the rangeloop checker.

package testdata

func RangeLoopTests() {
	var s []int
	for i, v := range s {
		go func() {
			println(i) // ERROR "range variable i enclosed by function"
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i, v := range s {
		defer func() {
			println(i) // ERROR "range variable i enclosed by function"
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i := range s {
		go func() {
			println(i) // ERROR "range variable i enclosed by function"
		}()
	}
	for _, v := range s {
		go func() {
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i, v := range s {
		go func() {
			println(i, v)
		}()
		println("unfortunately, we don't catch the error above because of this statement")
	}
	for i, v := range s {
		go func(i, v int) {
			println(i, v)
		}(i, v)
	}
	for i, v := range s {
		i, v := i, v
		go func() {
			println(i, v)
		}()
	}
	// If the key of the range statement is not an identifier
	// the code should not panic (it used to).
	var x [2]int
	var f int
	for x[0], f = range s {
		go func() {
			_ = f // ERROR "range variable f enclosed by function"
		}()
	}
}
