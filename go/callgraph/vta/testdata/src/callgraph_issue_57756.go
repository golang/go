// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

// Test that the values of a named function type are correctly
// flowing from interface objects i in i.Foo() to the receiver
// parameters of callees of i.Foo().

type H func()

func (h H) Do() {
	h()
}

type I interface {
	Do()
}

func Bar() I {
	return H(func() {})
}

func For(g G) {
	b := Bar()
	b.Do()

	g[0] = b
	g.Goo()
}

type G []I

func (g G) Goo() {
	g[0].Do()
}

// Relevant SSA:
// func Bar$1():
//   return
//
// func Bar() I:
//   t0 = changetype H <- func() (Bar$1)
//   t1 = make I <- H (t0)
//
// func For():
//   t0 = Bar()
//   t1 = invoke t0.Do()
//   t2 = &g[0:int]
//   *t2 = t0
//   t3 = (G).Goo(g)
//
// func (h H) Do():
//   t0 = h()
//
// func (g G) Goo():
//   t0 = &g[0:int]
//   t1 = *t0
//   t2 = invoke t1.Do()

// WANT:
// For: (G).Goo(g) -> G.Goo; Bar() -> Bar; invoke t0.Do() -> H.Do
// H.Do: h() -> Bar$1
// G.Goo: invoke t1.Do() -> H.Do
