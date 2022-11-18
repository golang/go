// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type A struct{}

func (a A) foo() {}

type I interface{ foo() }

func Baz(i I) {
	j := &i
	k := &j
	**k = A{}
	i.foo()
	(**k).foo()
}

// Relevant SSA:
// func Baz(i I):
//   t0 = new I (i)
//   *t0 = i
//   t1 = new *I (j)
//   *t1 = t0
//   t2 = *t1
// 	 t3 = make I <- A (struct{}{}:A)                                       I
//   *t2 = t3
//   t4 = *t0
//   t5 = invoke t4.foo()
//   t6 = *t1
//   t7 = *t6
//   t8 = invoke t7.foo()

// Flow chain showing that A reaches i.foo():
//   Constant(A) -> t3 -> t2 <-> PtrInterface(I) <-> t0 -> t4
// Flow chain showing that A reaches (**k).foo():
//	 Constant(A) -> t3 -> t2 <-> PtrInterface(I) <-> t6 -> t7

// WANT:
// Local(i) -> Local(t0)
// Local(t0) -> Local(t4), PtrInterface(testdata.I)
// PtrInterface(testdata.I) -> Local(t0), Local(t2), Local(t6)
// Local(t2) -> PtrInterface(testdata.I)
// Constant(testdata.A) -> Local(t3)
// Local(t3) -> Local(t2)
// Local(t6) -> Local(t7), PtrInterface(testdata.I)
