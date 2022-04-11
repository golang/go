// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo() I
}

type A struct {
	i int
	a *A
}

func (a *A) Foo() I {
	return a
}

type B **B

type C *D
type D *C

func Bar(a *A, b *B, c *C, d *D) {
	Baz(a)
	Baz(a.a)

	sink(*b)
	sink(*c)
	sink(*d)
}

func Baz(i I) {
	i.Foo()
}

func sink(i interface{}) {
	print(i)
}

// Relevant SSA:
// func Baz(i I):
//   t0 = invoke i.Foo()
//   return
//
// func Bar(a *A, b *B):
//   t0 = make I <- *A (a)
//   t1 = Baz(t0)
//   ...

// WANT:
// Bar: Baz(t0) -> Baz; Baz(t4) -> Baz; sink(t10) -> sink; sink(t13) -> sink; sink(t7) -> sink
// Baz: invoke i.Foo() -> A.Foo
