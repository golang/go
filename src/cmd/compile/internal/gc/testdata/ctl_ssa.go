// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test control flow

package main

// nor_ssa calculates NOR(a, b).
// It is implemented in a way that generates
// phi control values.
func nor_ssa(a, b bool) bool {
	var c bool
	if a {
		c = true
	}
	if b {
		c = true
	}
	if c {
		return false
	}
	return true
}

func testPhiControl() {
	tests := [...][3]bool{ // a, b, want
		{false, false, true},
		{true, false, false},
		{false, true, false},
		{true, true, false},
	}
	for _, test := range tests {
		a, b := test[0], test[1]
		got := nor_ssa(a, b)
		want := test[2]
		if want != got {
			print("nor(", a, ", ", b, ")=", want, " got ", got, "\n")
			failed = true
		}
	}
}

func emptyRange_ssa(b []byte) bool {
	for _, x := range b {
		_ = x
	}
	return true
}

func testEmptyRange() {
	if !emptyRange_ssa([]byte{}) {
		println("emptyRange_ssa([]byte{})=false, want true")
		failed = true
	}
}

var failed = false

func main() {
	testPhiControl()
	testEmptyRange()
	if failed {
		panic("failed")
	}
}
