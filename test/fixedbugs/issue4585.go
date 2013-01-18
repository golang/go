// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4585: comparisons and hashes process blank
// fields and padding in structs.

package main

import "unsafe"

// T is a structure with padding.
type T struct {
	A     int16
	B     int64
	C     int16
	D     int64
	Dummy [64]byte
}

// U is a structure with a blank field
type U struct {
	A, _, B int
	Dummy   [64]byte
}

// USmall is like U but the frontend will inline comparison
// instead of calling the generated eq function.
type USmall struct {
	A, _, B int32
}

func test1() {
	var a, b U
	m := make(map[U]int)
	copy((*[16]byte)(unsafe.Pointer(&a))[:], "hello world!")
	a.A, a.B = 1, 2
	b.A, b.B = 1, 2
	if a != b {
		panic("broken equality: a != b")
	}

	m[a] = 1
	m[b] = 2
	if len(m) == 2 {
		panic("broken hash: len(m) == 2")
	}
	if m[a] != 2 {
		panic("m[a] != 2")
	}
}

func test2() {
	var a, b T
	m := make(map[T]int)

	copy((*[16]byte)(unsafe.Pointer(&a))[:], "hello world!")
	a.A, a.B, a.C, a.D = 1, 2, 3, 4
	b.A, b.B, b.C, b.D = 1, 2, 3, 4

	if a != b {
		panic("broken equality: a != b")
	}

	m[a] = 1
	m[b] = 2
	if len(m) == 2 {
		panic("broken hash: len(m) == 2")
	}
	if m[a] != 2 {
		panic("m[a] != 2")
	}
}

func test3() {
	var a, b USmall
	copy((*[12]byte)(unsafe.Pointer(&a))[:], "hello world!")
	a.A, a.B = 1, 2
	b.A, b.B = 1, 2
	if a != b {
		panic("broken equality: a != b")
	}
}

func main() {
	test1()
	test2()
	test3()
}
