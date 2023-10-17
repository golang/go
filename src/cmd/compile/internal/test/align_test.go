// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test to make sure that equality functions (and hash
// functions) don't do unaligned reads on architectures
// that can't do unaligned reads. See issue 46283.

package test

import "testing"

type T1 struct {
	x          float32
	a, b, c, d int16 // memequal64
}
type T2 struct {
	x          float32
	a, b, c, d int32 // memequal128
}

type A2 [2]byte // eq uses a 2-byte load
type A4 [4]byte // eq uses a 4-byte load
type A8 [8]byte // eq uses an 8-byte load

//go:noinline
func cmpT1(p, q *T1) {
	if *p != *q {
		panic("comparison test wrong")
	}
}

//go:noinline
func cmpT2(p, q *T2) {
	if *p != *q {
		panic("comparison test wrong")
	}
}

//go:noinline
func cmpA2(p, q *A2) {
	if *p != *q {
		panic("comparison test wrong")
	}
}

//go:noinline
func cmpA4(p, q *A4) {
	if *p != *q {
		panic("comparison test wrong")
	}
}

//go:noinline
func cmpA8(p, q *A8) {
	if *p != *q {
		panic("comparison test wrong")
	}
}

func TestAlignEqual(t *testing.T) {
	cmpT1(&T1{}, &T1{})
	cmpT2(&T2{}, &T2{})

	m1 := map[T1]bool{}
	m1[T1{}] = true
	m1[T1{}] = false
	if len(m1) != 1 {
		t.Fatalf("len(m1)=%d, want 1", len(m1))
	}
	m2 := map[T2]bool{}
	m2[T2{}] = true
	m2[T2{}] = false
	if len(m2) != 1 {
		t.Fatalf("len(m2)=%d, want 1", len(m2))
	}

	type X2 struct {
		y byte
		z A2
	}
	var x2 X2
	cmpA2(&x2.z, &A2{})
	type X4 struct {
		y byte
		z A4
	}
	var x4 X4
	cmpA4(&x4.z, &A4{})
	type X8 struct {
		y byte
		z A8
	}
	var x8 X8
	cmpA8(&x8.z, &A8{})
}
