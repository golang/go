// errorcheck -0 -d=append,slice,ssa/prove/debug=1

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

// s1_if_false_then_anything
func s1_if_false_then_anything(x **[]int, xs **string, i, j int) {
	z := (**x)[0:i]
	z = z[i : i+1]
	println(z) // if we get here, then we have proven that i==i+1 (this cannot happen, but the program is still being analyzed...)

	zs := (**xs)[0:i] // since i=i+1 is proven, i+1 is "in bounds", ha-ha
	zs = zs[i : i+1]  // ERROR "Proved boolean IsSliceInBounds$"
	println(zs)
}

func s1(x **[]int, xs **string, i, j int) {
	var z []int
	z = (**x)[2:]
	z = (**x)[2:len(**x)] // ERROR "Proved boolean IsSliceInBounds$"
	z = (**x)[2:cap(**x)] // ERROR "Proved IsSliceInBounds$"
	z = (**x)[i:i]        // -ERROR "Proved IsSliceInBounds"
	z = (**x)[1:i:i]      // ERROR "Proved boolean IsSliceInBounds$"
	z = (**x)[i:j:0]
	z = (**x)[i:0:j] // ERROR "Disproved IsSliceInBounds$"
	z = (**x)[0:i:j] // ERROR "Proved boolean IsSliceInBounds$"
	z = (**x)[0:]    // ERROR "slice: omit slice operation$"
	z = (**x)[2:8]   // ERROR "Proved slicemask not needed$"
	println(z)
	z = (**x)[2:2]
	z = (**x)[0:i]
	z = (**x)[2:i:8] // ERROR "Disproved IsSliceInBounds$" "Proved IsSliceInBounds$"
	z = (**x)[i:2:i] // ERROR "Proved IsSliceInBounds$" "Proved boolean IsSliceInBounds$"

	z = z[0:i] // ERROR "Proved boolean IsSliceInBounds"
	z = z[0:i : i+1]
	z = z[i : i+1] // ERROR "Proved boolean IsSliceInBounds$"

	println(z)

	var zs string
	zs = (**xs)[2:]
	zs = (**xs)[2:len(**xs)] // ERROR "Proved IsSliceInBounds$" "Proved boolean IsSliceInBounds$"
	zs = (**xs)[i:i]         // -ERROR "Proved boolean IsSliceInBounds"
	zs = (**xs)[0:]          // ERROR "slice: omit slice operation$"
	zs = (**xs)[2:8]
	zs = (**xs)[2:2] // ERROR "Proved boolean IsSliceInBounds$"
	zs = (**xs)[0:i] // ERROR "Proved boolean IsSliceInBounds$"

	zs = zs[0:i]     // See s1_if_false_then_anything above to explain the counterfactual bounds check result below
	zs = zs[i : i+1] // ERROR "Proved boolean IsSliceInBounds$"
	println(zs)
}
