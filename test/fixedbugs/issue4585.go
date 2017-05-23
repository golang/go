// run

// Copyright 2013 The Go Authors. All rights reserved.
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

// V has padding but not on the first field.
type V struct {
	A1, A2, A3 int32
	B          int16
	C          int32
}

// W has padding at the end.
type W struct {
	A1, A2, A3 int32
	B          int32
	C          int8
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

func test4() {
	var a, b V
	m := make(map[V]int)

	copy((*[20]byte)(unsafe.Pointer(&a))[:], "Hello World, Gopher!")
	a.A1, a.A2, a.A3, a.B, a.C = 1, 2, 3, 4, 5
	b.A1, b.A2, b.A3, b.B, b.C = 1, 2, 3, 4, 5

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

func test5() {
	var a, b W
	m := make(map[W]int)

	copy((*[20]byte)(unsafe.Pointer(&a))[:], "Hello World, Gopher!")
	a.A1, a.A2, a.A3, a.B, a.C = 1, 2, 3, 4, 5
	b.A1, b.A2, b.A3, b.B, b.C = 1, 2, 3, 4, 5

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

func main() {
	test1()
	test2()
	test3()
	test4()
	test5()
}
