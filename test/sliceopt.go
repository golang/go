// +build !amd64
// errorcheck -0 -d=append,slice

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check optimization results for append and slicing.

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

func s1(x **[]int, xs **string, i, j int) {
	var z []int
	z = (**x)[2:]         // ERROR "slice: omit check for 2nd index"
	z = (**x)[2:len(**x)] // not yet: "slice: reuse len" "slice: omit check for 2nd index"
	z = (**x)[2:cap(**x)] // not yet: "slice: reuse cap" "slice: omit check for 2nd index"
	z = (**x)[i:i]        // ERROR "slice: reuse 1st index" "slice: omit check for 1st index" "slice: result len == 0"
	z = (**x)[1:i:i]      // ERROR "slice: reuse 2nd index" "slice: omit check for 2nd index" "slice: result cap == result len"
	z = (**x)[i:j:0]      // ERROR "slice: omit check for 3rd index"
	z = (**x)[i:0:j]      // ERROR "slice: omit check for 2nd index"
	z = (**x)[0:i:j]      // ERROR "slice: omit check for 1st index" "slice: skip base adjustment for 1st index 0"
	z = (**x)[0:]         // ERROR "slice: omit slice operation"
	z = (**x)[2:8]        // ERROR "slice: omit check for 1st index" "slice: result len == 6"
	z = (**x)[2:2]        // ERROR "slice: omit check for 1st index" "slice: result len == 0"
	z = (**x)[0:i]        // ERROR "slice: omit check for 1st index" "slice: skip base adjustment for 1st index 0"
	z = (**x)[2:i:8]      // ERROR "slice: result cap == 6"
	z = (**x)[i:2:i]      // ERROR "slice: reuse 1st index" "slice: result cap == 0" "slice: skip base adjustment for cap == 0"

	z = z[0:i]       // ERROR "slice: omit check for 1st index" "slice: result cap not computed" "slice: skip base adjustment for 1st index 0" "slice: len-only update"
	z = z[0:i : i+1] // ERROR "slice: omit check for 1st index" "slice: skip base adjustment for 1st index 0" "slice: len/cap-only update"
	z = z[i : i+1]

	println(z)

	var zs string
	zs = (**xs)[2:]          // ERROR "slice: omit check for 2nd index"
	zs = (**xs)[2:len(**xs)] // not yet: "slice: reuse len" "slice: omit check for 2nd index"
	zs = (**xs)[i:i]         // ERROR "slice: reuse 1st index" "slice: omit check for 1st index" "slice: result len == 0" "slice: skip base adjustment for string len == 0"
	zs = (**xs)[0:]          // ERROR "slice: omit slice operation"
	zs = (**xs)[2:8]         // ERROR "slice: omit check for 1st index" "slice: result len == 6"
	zs = (**xs)[2:2]         // ERROR "slice: omit check for 1st index" "slice: result len == 0" "slice: skip base adjustment for string len == 0"
	zs = (**xs)[0:i]         // ERROR "slice: omit check for 1st index" "slice: skip base adjustment for 1st index 0"

	zs = zs[0:i] // ERROR "slice: omit check for 1st index" "slice: skip base adjustment for 1st index 0" "slice: len-only update"
	zs = zs[i : i+1]
	println(zs)
}
