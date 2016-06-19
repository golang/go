// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests short circuiting.

package main

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

func testAnd(arg1, arg2, wantRes bool) { testShortCircuit("AND", arg1, arg2, and_ssa, arg1, wantRes) }
func testOr(arg1, arg2, wantRes bool)  { testShortCircuit("OR", arg1, arg2, or_ssa, !arg1, wantRes) }

func testShortCircuit(opName string, arg1, arg2 bool, fn func(bool, bool) bool, wantRightCall, wantRes bool) {
	rightCalled = false
	got := fn(arg1, arg2)
	if rightCalled != wantRightCall {
		println("failed for", arg1, opName, arg2, "; rightCalled=", rightCalled, "want=", wantRightCall)
		failed = true
	}
	if wantRes != got {
		println("failed for", arg1, opName, arg2, "; res=", got, "want=", wantRes)
		failed = true
	}
}

var failed = false

func main() {
	testAnd(false, false, false)
	testAnd(false, true, false)
	testAnd(true, false, false)
	testAnd(true, true, true)

	testOr(false, false, false)
	testOr(false, true, true)
	testOr(true, false, true)
	testOr(true, true, true)

	if failed {
		panic("failed")
	}
}
