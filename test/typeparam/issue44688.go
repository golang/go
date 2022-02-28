// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// derived & expanded from cmd/compile/internal/types2/testdata/fixedbugs/issue44688.go2

package main

type A1[T any] struct {
	val T
}

func (p *A1[T]) m1(val T) {
	p.val = val
}

type A2[T any] interface {
	m2(T)
}

type B1[T any] struct {
	filler int
	*A1[T]
	A2[T]
}

type B2[T any] interface {
	A2[T]
}

type ImpA2[T any] struct {
	f T
}

func (a2 *ImpA2[T]) m2(s T) {
	a2.f = s
}

type C[T any] struct {
	filler1 int
	filler2 int
	B1[T]
}

type D[T any] struct {
	filler1 int
	filler2 int
	filler3 int
	C[T]
}

func test1[T any](arg T) {
	// calling embedded methods
	var b1 B1[T]
	b1.A1 = &A1[T]{}
	b1.A2 = &ImpA2[T]{}

	b1.A1.m1(arg)
	b1.m1(arg)

	b1.A2.m2(arg)
	b1.m2(arg)

	var b2 B2[T]
	b2 = &ImpA2[T]{}
	b2.m2(arg)

	// a deeper nesting
	var d D[T]
	d.C.B1.A1 = &A1[T]{}
	d.C.B1.A2 = &ImpA2[T]{}
	d.m1(arg)
	d.m2(arg)

	// calling method expressions
	m1x := B1[T].m1
	m1x(b1, arg)
	// TODO(khr): reenable these.
	//m2x := B2[T].m2
	//m2x(b2, arg)

	// calling method values
	m1v := b1.m1
	m1v(arg)
	m2v := b1.m2
	m2v(arg)
	b2v := b2.m2
	b2v(arg)
}

func test2() {
	// calling embedded methods
	var b1 B1[string]
	b1.A1 = &A1[string]{}
	b1.A2 = &ImpA2[string]{}

	b1.A1.m1("")
	b1.m1("")

	b1.A2.m2("")
	b1.m2("")

	var b2 B2[string]
	b2 = &ImpA2[string]{}
	b2.m2("")

	// a deeper nesting
	var d D[string]
	d.C.B1.A1 = &A1[string]{}
	d.C.B1.A2 = &ImpA2[string]{}
	d.m1("")
	d.m2("")

	// calling method expressions
	m1x := B1[string].m1
	m1x(b1, "")
	m2x := B2[string].m2
	m2x(b2, "")

	// calling method values
	m1v := b1.m1
	m1v("")
	m2v := b1.m2
	m2v("")
	b2v := b2.m2
	b2v("")
}

// actual test case from issue

type A[T any] struct{}

func (*A[T]) f(T) {}

type B[T any] struct{ A[T] }

func test3() {
	var b B[string]
	b.A.f("")
	b.f("")
}

func main() {
	test1[string]("")
	test2()
	test3()
}
