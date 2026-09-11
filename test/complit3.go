// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that struct literal assignment to slice elements works correctly
// when the RHS contains function calls that grow the LHS slice.
// The order pass evaluates the LHS index after extracting RHS side effects,
// so the slice is already grown when the index is evaluated at runtime.

package main

type S struct {
	A int
	B string
	C float64
	D int64
	E bool
	F string
	G int
	H string
}

//go:noinline
func growSlice(sp *[]S) int {
	*sp = append(*sp, S{})
	return 99
}

func main() {
	testLocalClosure()
	testNoinlineGrow()
}

func testLocalClosure() {
	// s starts empty (len=0). The closure in the RHS grows it to len=1
	// via append before returning. The order pass extracts the closure
	// call, so by the time s[0] is evaluated, len(s)==1.
	s := []S{}
	s[0] = S{
		A: func() int { s = append(s, S{}); return 42 }(),
		B: "hello",
		C: 3.14,
		D: 100,
		E: true,
		F: "world",
		G: 7,
		H: "test",
	}
	if s[0].A != 42 {
		panic("testLocalClosure: s[0].A != 42")
	}
	if s[0].B != "hello" {
		panic("testLocalClosure: s[0].B != hello")
	}
	if s[0].G != 7 {
		panic("testLocalClosure: s[0].G != 7")
	}
	if len(s) != 1 {
		panic("testLocalClosure: len(s) != 1")
	}
}

func testNoinlineGrow() {
	// s starts empty. growSlice appends a zero S{} and returns 99.
	// The order pass extracts the call to an autotemp, so s has
	// len=1 by the time s[0] is evaluated.
	s := make([]S, 0)
	s[0] = S{
		A: growSlice(&s),
		B: "hello",
		C: 3.14,
		D: 100,
		E: true,
		F: "world",
		G: 7,
		H: "test",
	}
	if s[0].A != 99 {
		panic("testNoinlineGrow: s[0].A != 99")
	}
	if s[0].B != "hello" {
		panic("testNoinlineGrow: s[0].B != hello")
	}
	if s[0].H != "test" {
		panic("testNoinlineGrow: s[0].H != test")
	}
	if len(s) != 1 {
		panic("testNoinlineGrow: len(s) != 1")
	}
}
