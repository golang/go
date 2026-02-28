// errorcheck -0 -d=append,slice

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check optimization results for append and slicing.

package main

func a1(x []int, y int) []int {
	x = append(x, y) // ERROR "append: len-only update \(in local slice\)$"
	return x
}

func a2(x []int, y int) []int {
	return append(x, y)
}

func a3(x *[]int, y int) {
	*x = append(*x, y) // ERROR "append: len-only update$"
}

func s1(x **[]int, xs **string, i, j int) {
	var z []int
	z = (**x)[0:] // ERROR "slice: omit slice operation$"
	println(z)

	var zs string
	zs = (**xs)[0:] // ERROR "slice: omit slice operation$"
	println(zs)
}
