// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type A struct{}

func (a A) foo() {}

func Bar() {}

func Baz(a A) {
	a.foo()
	Bar()
	Baz(A{})
}

// Relevant SSA:
// func Baz(a A):
//   t0 = (A).foo(a)
//   t1 = Bar()
//   t2 = Baz(struct{}{}:A)

// WANT:
// Baz: (A).foo(a) -> A.foo; Bar() -> Bar; Baz(struct{}{}:A) -> Baz
