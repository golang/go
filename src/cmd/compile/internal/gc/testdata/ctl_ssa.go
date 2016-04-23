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

func switch_ssa(a int) int {
	ret := 0
	switch a {
	case 5:
		ret += 5
	case 4:
		ret += 4
	case 3:
		ret += 3
	case 2:
		ret += 2
	case 1:
		ret += 1
	}
	return ret

}

func fallthrough_ssa(a int) int {
	ret := 0
	switch a {
	case 5:
		ret++
		fallthrough
	case 4:
		ret++
		fallthrough
	case 3:
		ret++
		fallthrough
	case 2:
		ret++
		fallthrough
	case 1:
		ret++
	}
	return ret

}

func testFallthrough() {
	for i := 0; i < 6; i++ {
		if got := fallthrough_ssa(i); got != i {
			println("fallthrough_ssa(i) =", got, "wanted", i)
			failed = true
		}
	}
}

func testSwitch() {
	for i := 0; i < 6; i++ {
		if got := switch_ssa(i); got != i {
			println("switch_ssa(i) =", got, "wanted", i)
			failed = true
		}
	}
}

type junk struct {
	step int
}

// flagOverwrite_ssa is intended to reproduce an issue seen where a XOR
// was scheduled between a compare and branch, clearing flags.
//go:noinline
func flagOverwrite_ssa(s *junk, c int) int {
	if '0' <= c && c <= '9' {
		s.step = 0
		return 1
	}
	if c == 'e' || c == 'E' {
		s.step = 0
		return 2
	}
	s.step = 0
	return 3
}

func testFlagOverwrite() {
	j := junk{}
	if got := flagOverwrite_ssa(&j, ' '); got != 3 {
		println("flagOverwrite_ssa =", got, "wanted 3")
		failed = true
	}
}

var failed = false

func main() {
	testPhiControl()
	testEmptyRange()

	testSwitch()
	testFallthrough()

	testFlagOverwrite()

	if failed {
		panic("failed")
	}
}
