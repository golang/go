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
//   ...
//   t2 = (A).foo(t1)
//   t3 = Bar()
//   ...
//   t6 = Baz(t5)

// WANT:
// Baz: (A).foo(t1) -> A.foo; Bar() -> Bar; Baz(t5) -> Baz
