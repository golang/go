// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type A struct{}

func (a A) Foo() {}

type B struct{}

func (b B) Foo() {}

type C struct{}

func (c C) Foo() {}

func NewB() B {
	return B{}
}

func Do(b bool) I {
	if b {
		return A{}
	}

	c := C{}
	c.Foo()

	return NewB()
}

func Baz(b bool) {
	Do(b).Foo()
}

// Relevant SSA:
// func Baz(b bool):
//   t0 = Do(b)
//   t1 = invoke t0.Foo()
//   return

// func Do(b bool) I:
//    ...
//   t3 = local C (c)
//   t4 = *t3
//   t5 = (C).Foo(t4)
//   t6 = NewB()
//   t7 = make I <- B (t6)
//   return t7

// WANT:
// Baz: Do(b) -> Do; invoke t0.Foo() -> A.Foo, B.Foo
// Do: (C).Foo(t4) -> C.Foo; NewB() -> NewB
