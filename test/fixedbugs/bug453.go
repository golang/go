// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4138: bug in floating-point registers numbering.
// Makes 6g unable to use more than 11 registers.

package main

func formula() float32 {
	mA := [1]float32{1.0}
	det1 := mA[0]
	det2 := mA[0]
	det3 := mA[0]
	det4 := mA[0]
	det5 := mA[0]
	det6 := mA[0]
	det7 := mA[0]
	det8 := mA[0]
	det9 := mA[0]
	det10 := mA[0]
	det11 := mA[0]
	det12 := mA[0]

	return det1 + det2*det3 +
		det4*det5 + det6*det7 +
		det8*det9 + det10*det11 +
		det12
}

func main() {
	x := formula()
	if x != 7.0 {
		println(x, 7.0)
		panic("x != 7.0")
	}
}
