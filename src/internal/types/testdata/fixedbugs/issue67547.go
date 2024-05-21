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
