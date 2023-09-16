// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 9083: map/chan error messages show non-explicit capacity.

package main

// untyped constant
const zero = 0

func main() {
	var x int
	_ = x
	x = make(map[int]int)       // ERROR "cannot use make\(map\[int\]int\)|incompatible"
	x = make(map[int]int, 0)    // ERROR "cannot use make\(map\[int\]int, 0\)|incompatible"
	x = make(map[int]int, zero) // ERROR "cannot use make\(map\[int\]int, zero\)|incompatible"
	x = make(chan int)          // ERROR "cannot use make\(chan int\)|incompatible"
	x = make(chan int, 0)       // ERROR "cannot use make\(chan int, 0\)|incompatible"
	x = make(chan int, zero)    // ERROR "cannot use make\(chan int, zero\)|incompatible"
}
