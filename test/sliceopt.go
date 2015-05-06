// errorcheck -0 -d=append

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check optimization results for append.

package main

func a1(x []int, y int) []int {
	x = append(x, y) // ERROR "append: len-only update"
	return x
}

func a2(x []int, y int) []int {
	return append(x, y) // ERROR "append: full update"
}

func a3(x *[]int, y int) {
	*x = append(*x, y) // ERROR "append: len-only update"
}
