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

func Do(i **I) {
	**i = A{}
}

func Bar(i **I) {
	**i = B{}
}

func Baz(i **I) {
	Do(i)
	(**i).Foo()
}

// Relevant SSA:
//  func Baz(i **I):
//   t0 = Do(i)
//   t1 = *i
//   t2 = *t1
//   t3 = invoke t2.Foo()
//   return

//  func Bar(i **I):
//   t0 = *i
//   t1 = local B (complit)
//   t2 = *t1
//   t3 = make I <- B (t2)
//   *t0 = t3
//   return

// func Do(i **I):
//   t0 = *i
//   t1 = local A (complit)
//   t2 = *t1
//   t3 = make I <- A (t2)
//   *t0 = t3
//   return

// WANT:
// Baz: Do(i) -> Do; invoke t2.Foo() -> A.Foo, B.Foo
