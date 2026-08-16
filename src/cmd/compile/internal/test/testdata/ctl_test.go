// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test control flow

package main

import "testing"

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

func testPhiControl(t *testing.T) {
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
			t.Errorf("nor(%t, %t)=%t got %t", a, b, want, got)
		}
	}
}

func emptyRange_ssa(b []byte) bool {
	for _, x := range b {
		_ = x
	}
	return true
}

func testEmptyRange(t *testing.T) {
	if !emptyRange_ssa([]byte{}) {
		t.Errorf("emptyRange_ssa([]byte{})=false, want true")
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

func switch_nonconstcase(n, c int) int {
	switch n {
	case 1:
		return 1
	case 2:
		return 2
	case c:
		return 9
	case 3:
		return 3
	case 4:
		return 4
	}
	return 0
}

func switchAssign_ssa(a int) (int, int) {
	ret := -1
	switch a {
	case 1:
		ret = 10
	case 2:
		ret = 20
	case 3:
		ret = a * 10
	case 4:
		ret = 40
	case 6:
		ret = 60
	default:
		ret = 99
	}
	return ret, a
}

func switchAssignLocal_ssa(a int) int {
	ret := -1
	switch a {
	case 1:
		ret = 10
	case 2:
		ret = 20
	case 3:
		ret = a * 10
	case 4:
		ret = 40
	case 6:
		ret = 60
	default:
		ret = 99
	}
	return ret + 1
}

func switchAssignReturnDefault_ssa(a int) int {
	ret := -1
	switch a {
	case 1:
		ret = 10
	case 2:
		ret = 20
	case 4:
		ret = 40
	case 5:
		ret = 50
	default:
		return 99
	}
	return ret + 1
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

func testFallthrough(t *testing.T) {
	for i := 0; i < 6; i++ {
		if got := fallthrough_ssa(i); got != i {
			t.Errorf("fallthrough_ssa(i) = %d, wanted %d", got, i)
		}
	}
}

func testSwitch(t *testing.T) {
	for i := 0; i < 6; i++ {
		if got := switch_ssa(i); got != i {
			t.Errorf("switch_ssa(i) = %d, wanted %d", got, i)
		}
	}
	if got := switch_nonconstcase(3, 3); got != 9 {
		t.Errorf("switch_nonconstcase(3, 3) = %d, wanted 9", got)
	}
	want := [...]int{99, 10, 20, 30, 40, 99, 60, 99}
	for i, want := range want {
		if got, _ := switchAssign_ssa(i); got != want {
			t.Errorf("switchAssign_ssa(%d) = %d, wanted %d", i, got, want)
		}
		if got := switchAssignLocal_ssa(i); got != want+1 {
			t.Errorf("switchAssignLocal_ssa(%d) = %d, wanted %d", i, got, want+1)
		}
	}
	if got := switchAssignReturnDefault_ssa(3); got != 99 {
		t.Errorf("switchAssignReturnDefault_ssa(3) = %d, wanted 99", got)
	}
}

type junk struct {
	step int
}

// flagOverwrite_ssa is intended to reproduce an issue seen where a XOR
// was scheduled between a compare and branch, clearing flags.
//
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

func testFlagOverwrite(t *testing.T) {
	j := junk{}
	if got := flagOverwrite_ssa(&j, ' '); got != 3 {
		t.Errorf("flagOverwrite_ssa = %d, wanted 3", got)
	}
}

func TestCtl(t *testing.T) {
	testPhiControl(t)
	testEmptyRange(t)

	testSwitch(t)
	testFallthrough(t)

	testFlagOverwrite(t)
}
