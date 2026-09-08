// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 81266: check that we generate correct code when
// using the subroutine mechanism in the equality/hash generator.

package main

import (
	"fmt"
	"unsafe"
)

func check(name string, got, want bool) {
	if got != want {
		panic(fmt.Sprintf("%s: got %v want %v", name, got, want))
	}
}

// Big is large enough for subroutine treatment, and has trailing padding.
type Big struct {
	Filler [2000]byte
	S      string
	B      byte
}

// Outer embeds Big as a non-top-level field, and has a field after
// it. There is padding between the two fields.
type Outer struct {
	Big Big
	X   byte
}

// fill sets every byte of the size bytes at p to b. Used to poison
// padding bytes with garbage, so that tests which pass despite
// differing padding prove the padding is correctly ignored.
func fill(p unsafe.Pointer, size uintptr, b byte) {
	buf := unsafe.Slice((*byte)(p), size)
	for i := range buf {
		buf[i] = b
	}
}

var filler [2000]byte

func init() {
	for i := range filler {
		filler[i] = byte(i)
	}
}

func mkBig(s string, b byte, pad byte) Big {
	var x Big
	fill(unsafe.Pointer(&x), unsafe.Sizeof(x), pad)
	x.Filler = filler
	x.S = s
	x.B = b
	return x
}

func testBasic() {
	a := Outer{Big: mkBig("foo", 7, 0xAA), X: 1}
	c := Outer{Big: mkBig("foo", 7, 0x55), X: 1}
	check("equal despite differing padding inside Big", a == c, true)

	d := Outer{Big: mkBig("foo", 7, 0xAA), X: 2}
	check("differ in trailing field X", a == d, false)

	e := Outer{Big: mkBig("bar", 7, 0xAA), X: 1}
	check("differ in Big.S", a == e, false)

	g := Outer{Big: mkBig("foo", 9, 0xAA), X: 1}
	check("differ in Big.B", a == g, false)

	h := a
	h.Big.Filler[1000] ^= 0xFF
	check("differ in Big.Filler", a == h, false)
}

// testPadding makes sure different paddings don't prevent equality.
func testPadding() {
	var a, c Outer
	fill(unsafe.Pointer(&a), unsafe.Sizeof(a), 0x11)
	fill(unsafe.Pointer(&c), unsafe.Sizeof(c), 0x99)
	a.Big.Filler = filler
	c.Big.Filler = filler
	a.Big.S = "padding-check"
	c.Big.S = "padding-check"
	a.Big.B = 42
	c.Big.B = 42
	a.X = 5
	c.X = 5
	check("equal with garbage-filled padding", a == c, true)
}

// testHash exercises the hash side of the subroutine mechanism.
func testHash() {
	m := make(map[Outer]int)
	k1 := Outer{Big: mkBig("k1", 1, 0), X: 1}
	k2 := Outer{Big: mkBig("k2", 2, 0), X: 2}
	m[k1] = 100
	m[k2] = 200

	look1 := Outer{Big: mkBig("k1", 1, 0xFF), X: 1} // differing padding
	look2 := Outer{Big: mkBig("k2", 2, 0xFF), X: 2}
	v1, ok1 := m[look1]
	v2, ok2 := m[look2]
	check("map lookup k1 found", ok1, true)
	check("map lookup k2 found", ok2, true)
	if v1 != 100 {
		panic(fmt.Sprintf("map k1 value: got %v want 100", v1))
	}
	if v2 != 200 {
		panic(fmt.Sprintf("map k2 value: got %v want 200", v2))
	}

	notFound := Outer{Big: mkBig("k1", 1, 0), X: 99}
	_, ok3 := m[notFound]
	check("map lookup miss", ok3, false)
}

// testNested exercises two levels of subroutine nesting
// (Outer2 -> Level1 -> Big).
func testNested() {
	// Level1 embeds Big, so it becomes a subroutine when embedded
	// in Outer2. This produces two levels of subroutine nesting.
	type Level1 struct {
		B  Big
		F2 [2000]byte
		T  string
		C  byte
	}

	type Outer2 struct {
		L1 Level1
		Y  byte
	}

	mkLevel1 := func(s1, s2 string, b1, b2 byte) Level1 {
		var l Level1
		l.B = mkBig(s1, b1, 0)
		l.F2 = filler
		l.T = s2
		l.C = b2
		return l
	}

	a := Outer2{L1: mkLevel1("inner1", "outer1", 1, 2), Y: 9}
	c := Outer2{L1: mkLevel1("inner1", "outer1", 1, 2), Y: 9}
	check("nested equal", a == c, true)

	d := a
	d.L1.B.S = "changed"
	check("nested differ in deepest field", a == d, false)

	e := a
	e.L1.T = "changed"
	check("nested differ in Level1 field", a == e, false)

	g := a
	g.Y = 10
	check("nested differ in outer field", a == g, false)

	mm := make(map[Outer2]string)
	mm[a] = "a-value"
	if got := mm[c]; got != "a-value" {
		panic(fmt.Sprintf("nested map lookup: got %q want %q", got, "a-value"))
	}
}

// testArray exercises an array of large elements, each of which uses
// the subroutine mechanism.
func testArray() {
	type BigArr struct {
		A [3]Big
	}
	var a, c BigArr
	for i := range a.A {
		a.A[i] = mkBig(fmt.Sprintf("elem%d", i), byte(i), byte(i))
		c.A[i] = mkBig(fmt.Sprintf("elem%d", i), byte(i), byte(i+100)) // different padding
	}
	check("array equal despite differing padding", a == c, true)

	d := a
	d.A[2].S = "different"
	check("array differ in one element", a == d, false)
}

// testBoundary checks types right at the maxExpandSize (1024) cutoff,
// on both sides.
func testBoundary() {
	type Exactly1024 struct {
		F [1024 - 16]byte // + string(16) = 1024 exactly, on a 64-bit platform
		S string
	}
	type Container1024 struct {
		V Exactly1024
		X byte
	}
	var a, c Container1024
	a.V.S = "same"
	c.V.S = "same"
	a.X = 1
	c.X = 1
	check("boundary <=1024 equal", a == c, true)
	d := a
	d.V.S = "diff"
	check("boundary <=1024 differ", a == d, false)

	type Exactly1025 struct {
		F [1025 - 16]byte
		S string
	}
	type Container1025 struct {
		V Exactly1025
		X byte
	}
	var e, g Container1025
	e.V.S = "same"
	g.V.S = "same"
	e.X = 1
	g.X = 1
	check("boundary >1024 equal", e == g, true)
	h := e
	h.V.S = "diff"
	check("boundary >1024 differ", e == h, false)
}

func main() {
	testBasic()
	testPadding()
	testHash()
	testNested()
	testArray()
	testBoundary()
}
