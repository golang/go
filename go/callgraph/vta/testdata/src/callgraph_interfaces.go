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
//   t1 = (C).Foo(struct{}{}:C)
//   t2 = NewB()
//   t3 = make I <- B (t2)
//   return t3

// WANT:
// Baz: Do(b) -> Do; invoke t0.Foo() -> A.Foo, B.Foo
// Do: (C).Foo(struct{}{}:C) -> C.Foo; NewB() -> NewB
