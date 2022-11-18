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

// The command a.f = &i introduces aliasing that results in
// A and B reaching both *A.f and return value of Do(a, b, c).

// WANT:
// Baz: Do(a, t0, c) -> Do; invoke t2.Foo() -> A.Foo, B.Foo
// Do: invoke t8.Foo() -> A.Foo, B.Foo
