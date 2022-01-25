// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface{}

func Bar(ii I) (I, I) {
	return Foo(ii)
}

func Foo(iii I) (I, I) {
	return iii, iii
}

func Do(j I) *I {
	return &j
}

func Baz(i I) *I {
	Bar(i)
	return Do(i)
}

// Relevant SSA:
// func Bar(ii I) (I, I):
//   t0 = Foo(ii)
//   t1 = extract t0 #0
//   t2 = extract t0 #1
//   return t1, t2
//
// func Foo(iii I) (I, I):
//   return iii, iii
//
// func Do(j I) *I:
//   t0 = new I (j)
//   *t0 = j
//   return t0
//
// func Baz(i I):
//   t0 = Bar(i)
//   t1 = Do(i)
//   return t1

// t0 and t1 in the last edge correspond to the nodes
// of Do and Baz. This edge is induced by Do(i).

// WANT:
// Local(i) -> Local(ii), Local(j)
// Local(ii) -> Local(iii)
// Local(iii) -> Local(t0[0]), Local(t0[1])
// Local(t1) -> Local(t0[0])
// Local(t2) -> Local(t0[1])
// Local(t0) -> Local(t1)
