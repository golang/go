// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type A struct {
	f *I
}

func (a A) Foo() {}

type B struct{}

func (b B) Foo() {}

func Do(a A, i I, c bool) *I {
	if c {
		*a.f = a
	} else {
		a.f = &i
	}
	(*a.f).Foo()
	return &i
}

func Baz(a A, b B, c bool) {
	x := Do(a, b, c)
	(*x).Foo()
}

// Relevant SSA:
// func Baz(a A, b B, c bool):
//   t0 = local A (a)
//   ...
//   t5 = Do(t2, t4, c)
//   t6 = *t5
//   t7 = invoke t6.Foo()
//   return

// func Do(a A, i I, c bool) *I:
//   t0 = local A (a)
//   *t0 = a
//   ...
//   if c goto 1 else 3
//  1:
//   t2 = &t0.f [#0]
//   ...
//   jump 2
//  2:
//   t6 = &t0.f [#0]
//   ...
//   t9 = invoke t8.Foo()
//   return t1
//  3:
//   t10 = &t0.f [#0]      alias between A.f and t10
//   *t10 = t1             alias between t10 and t1
//   jump 2

// The command a.f = &i introduces aliasing that results in
// A and B reaching both *A.f and return value of Do(a, b, c).

// WANT:
// Baz: Do(t2, t4, c) -> Do; invoke t6.Foo() -> A.Foo, B.Foo
// Do: invoke t8.Foo() -> A.Foo, B.Foo
