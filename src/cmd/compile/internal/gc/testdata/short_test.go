// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests short circuiting.

package main

import "testing"

func and_ssa(arg1, arg2 bool) bool {
	return arg1 && rightCall(arg2)
}

func or_ssa(arg1, arg2 bool) bool {
	return arg1 || rightCall(arg2)
}

var rightCalled bool

//go:noinline
func rightCall(v bool) bool {
	rightCalled = true
	return v
	panic("unreached")
}

func testAnd(t *testing.T, arg1, arg2, wantRes bool) {
	testShortCircuit(t, "AND", arg1, arg2, and_ssa, arg1, wantRes)
}
func testOr(t *testing.T, arg1, arg2, wantRes bool) {
	testShortCircuit(t, "OR", arg1, arg2, or_ssa, !arg1, wantRes)
}

func testShortCircuit(t *testing.T, opName string, arg1, arg2 bool, fn func(bool, bool) bool, wantRightCall, wantRes bool) {
	rightCalled = false
	got := fn(arg1, arg2)
	if rightCalled != wantRightCall {
		t.Errorf("failed for %t %s %t; rightCalled=%t want=%t", arg1, opName, arg2, rightCalled, wantRightCall)
	}
	if wantRes != got {
		t.Errorf("failed for %t %s %t; res=%t want=%t", arg1, opName, arg2, got, wantRes)
	}
}

// TestShortCircuit tests OANDAND and OOROR expressions and short circuiting.
func TestShortCircuit(t *testing.T) {
	testAnd(t, false, false, false)
	testAnd(t, false, true, false)
	testAnd(t, true, false, false)
	testAnd(t, true, true, true)

	testOr(t, false, false, false)
	testOr(t, false, true, true)
	testOr(t, true, false, true)
	testOr(t, true, true, true)
}
