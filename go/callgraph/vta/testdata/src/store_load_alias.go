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
//   t3 = local A (complit)
//   t4 = *t3
//   t5 = make I <- A (t4)
//   *t2 = t5
//   t6 = *t0
//   t7 = invoke t6.foo()
//   t8 = *t1
//   t9 = *t8
//   t10 = invoke t9.foo()

// Flow chain showing that A reaches i.foo():
//   t4 (A) -> t5 -> t2 <-> PtrInterface(I) <-> t0 -> t6
// Flow chain showing that A reaches (**k).foo():
//	 t4 (A) -> t5 -> t2 <-> PtrInterface(I) <-> t8 -> t9

// WANT:
// Local(i) -> Local(t0)
// Local(t0) -> Local(t6), PtrInterface(testdata.I)
// PtrInterface(testdata.I) -> Local(t0), Local(t2), Local(t8)
// Local(t2) -> PtrInterface(testdata.I)
// Local(t4) -> Local(t5)
// Local(t5) -> Local(t2)
// Local(t8) -> Local(t9), PtrInterface(testdata.I)
