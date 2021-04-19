// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package P

type A1[T any] struct{}

func (*A1[T]) m1(T) {}

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

func _() {
	// calling embedded methods
	var b1 B1[string]

	b1.A1.m1("")
	b1.m1("")

	b1.A2.m2("")
	b1.m2("")

	var b2 B2[string]
	b2.m2("")

	// a deeper nesting
	var d D[string]
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

func _() {
	var b B[string]
	b.A.f("")
	b.f("")
}
