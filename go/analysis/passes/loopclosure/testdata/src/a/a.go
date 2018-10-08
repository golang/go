// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the loopclosure checker.

package testdata

func _() {
	var s []int
	for i, v := range s {
		go func() {
			println(i) // want "loop variable i captured by func literal"
			println(v) // want "loop variable v captured by func literal"
		}()
	}
	for i, v := range s {
		defer func() {
			println(i) // want "loop variable i captured by func literal"
			println(v) // want "loop variable v captured by func literal"
		}()
	}
	for i := range s {
		go func() {
			println(i) // want "loop variable i captured by func literal"
		}()
	}
	for _, v := range s {
		go func() {
			println(v) // want "loop variable v captured by func literal"
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
			_ = f // want "loop variable f captured by func literal"
		}()
	}
	type T struct {
		v int
	}
	for _, v := range s {
		go func() {
			_ = T{v: 1}
			_ = map[int]int{v: 1} // want "loop variable v captured by func literal"
		}()
	}

	// ordinary for-loops
	for i := 0; i < 10; i++ {
		go func() {
			print(i) // want "loop variable i captured by func literal"
		}()
	}
	for i, j := 0, 1; i < 100; i, j = j, i+j {
		go func() {
			print(j) // want "loop variable j captured by func literal"
		}()
	}
	type cons struct {
		car int
		cdr *cons
	}
	var head *cons
	for p := head; p != nil; p = p.cdr {
		go func() {
			print(p.car) // want "loop variable p captured by func literal"
		}()
	}
}
