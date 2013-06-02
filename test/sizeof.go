// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

type T struct {
	X int
}

var t T

func isUintptr(uintptr) {}

type T2 struct {
	A int32
	U2
}

type U2 struct {
	B int32
	C int32
}

var t2 T2
var p2 *T2

func main() {
	// Test unsafe.Sizeof, unsafe.Alignof, and unsafe.Offsetof all return uintptr.
	isUintptr(unsafe.Sizeof(t))
	isUintptr(unsafe.Alignof(t))
	isUintptr(unsafe.Offsetof(t.X))

	// Test correctness of Offsetof with respect to embedded fields (issue 4909).
	if unsafe.Offsetof(t2.C) != 8 {
		println(unsafe.Offsetof(t2.C), "!= 8")
		panic("unsafe.Offsetof(t2.C) != 8")
	}
	if unsafe.Offsetof(p2.C) != 8 {
		println(unsafe.Offsetof(p2.C), "!= 8")
		panic("unsafe.Offsetof(p2.C) != 8")
	}
	if unsafe.Offsetof(t2.U2.C) != 4 {
		println(unsafe.Offsetof(t2.U2.C), "!= 4")
		panic("unsafe.Offsetof(t2.U2.C) != 4")
	}
	if unsafe.Offsetof(p2.U2.C) != 4 {
		println(unsafe.Offsetof(p2.U2.C), "!= 4")
		panic("unsafe.Offsetof(p2.U2.C) != 4")
	}
	testDeep()
	testNotEmbedded()
}

type (
	S1 struct {
		A int64
		S2
	}
	S2 struct {
		B int64
		S3
	}
	S3 struct {
		C int64
		S4
	}
	S4 struct {
		D int64
		S5
	}
	S5 struct {
		E int64
		S6
	}
	S6 struct {
		F int64
		S7
	}
	S7 struct {
		G int64
		S8
	}
	S8 struct {
		H int64
		*S1
	}
)

func testDeep() {
	var s1 S1
	switch {
	case unsafe.Offsetof(s1.A) != 0:
		panic("unsafe.Offsetof(s1.A) != 0")
	case unsafe.Offsetof(s1.B) != 8:
		panic("unsafe.Offsetof(s1.B) != 8")
	case unsafe.Offsetof(s1.C) != 16:
		panic("unsafe.Offsetof(s1.C) != 16")
	case unsafe.Offsetof(s1.D) != 24:
		panic("unsafe.Offsetof(s1.D) != 24")
	case unsafe.Offsetof(s1.E) != 32:
		panic("unsafe.Offsetof(s1.E) != 32")
	case unsafe.Offsetof(s1.F) != 40:
		panic("unsafe.Offsetof(s1.F) != 40")
	case unsafe.Offsetof(s1.G) != 48:
		panic("unsafe.Offsetof(s1.G) != 48")
	case unsafe.Offsetof(s1.H) != 56:
		panic("unsafe.Offsetof(s1.H) != 56")
	case unsafe.Offsetof(s1.S1) != 64:
		panic("unsafe.Offsetof(s1.S1) != 64")
	case unsafe.Offsetof(s1.S1.S2.S3.S4.S5.S6.S7.S8.S1.S2) != 8:
		panic("unsafe.Offsetof(s1.S1.S2.S3.S4.S5.S6.S7.S8.S1.S2) != 8")
	}
}

func testNotEmbedded() {
	type T2 struct {
		B int32
		C int32
	}
	type T1 struct {
		A int32
		T2
	}
	type T struct {
		Dummy int32
		F     T1
		P     *T1
	}

	var t T
	var p *T
	switch {
	case unsafe.Offsetof(t.F.B) != 4:
		panic("unsafe.Offsetof(t.F.B) != 4")
	case unsafe.Offsetof(t.F.C) != 8:
		panic("unsafe.Offsetof(t.F.C) != 8")

	case unsafe.Offsetof(t.P.B) != 4:
		panic("unsafe.Offsetof(t.P.B) != 4")
	case unsafe.Offsetof(t.P.C) != 8:
		panic("unsafe.Offsetof(t.P.C) != 8")

	case unsafe.Offsetof(p.F.B) != 4:
		panic("unsafe.Offsetof(p.F.B) != 4")
	case unsafe.Offsetof(p.F.C) != 8:
		panic("unsafe.Offsetof(p.F.C) != 8")

	case unsafe.Offsetof(p.P.B) != 4:
		panic("unsafe.Offsetof(p.P.B) != 4")
	case unsafe.Offsetof(p.P.C) != 8:
		panic("unsafe.Offsetof(p.P.C) != 8")
	}
}
