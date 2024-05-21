// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P int]() {
	type A = P
	_ = A(0) // don't crash with this conversion
}

func _[P []int]() {
	type A = P
	_ = make(A, 10) // don't report an error for A
}

func _[P string]() {
	var t []byte
	type A = P
	var s A
	copy(t, s) // don't report an error for s
}

func _[P map[int]int]() {
	type A = P
	var m A
	clear(m) // don't report an error for m
}

type S1 struct {
	x int "S1.x"
}

type S2 struct {
	x int "S2.x"
}

func _[P1 S1, P2 S2]() {
	type A = P1
	var p A
	_ = P2(p) // conversion must be valid
}

func _[P1 S1, P2 S2]() {
	var p P1
	type A = P2
	_ = A(p) // conversion must be valid
}

func _[P int | string]() {
	var p P
	type A = int
	// preserve target type name A in error messages when using Alias types
	// (test are run with and without Alias types enabled, so we need to
	// keep both A and int in the error message)
	_ = A(p /* ERRORx `cannot convert string \(in P\) to type (A|int)` */)
}

func _[P struct{ x int }]() {
	var x struct{ x int }
	type A = P
	var _ A = x // assignment must be valid
}

func _[P struct{ x int }]() {
	type A = P
	var x A
	var _ struct{ x int } = x // assignment must be valid
}

func _[P []int | struct{}]() {
	type A = []int
	var a A
	var p P
	// preserve target type name A in error messages when using Alias types
	a = p // ERRORx `cannot assign struct{} \(in P\) to (A|\[\]int)`
	_ = a
}

// Test case for go.dev/issue/67540.
func _() {
	type (
		S struct{}
		A = *S
		T S
	)
	var p A
	_ = (*T)(p)
}
