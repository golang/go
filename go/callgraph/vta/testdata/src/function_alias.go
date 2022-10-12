// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type Doer func()

type A struct {
	foo func()
	do  Doer
}

func Baz(f func()) {
	j := &f
	k := &j
	**k = func() {}
	a := A{}
	a.foo = **k
	a.foo()
	a.do = a.foo
	a.do()
}

// Relevant SSA:
// func Baz(f func()):
//        t0 = new func() (f)
//        *t0 = f
//        t1 = new *func() (j)
//        *t1 = t0
//        t2 = *t1
//        *t2 = Baz$1
//        t3 = local A (a)
//        t4 = *t1
//        t5 = *t4
//        t6 = &t3.foo [#0]
//        *t6 = t5
//        t7 = &t3.foo [#0]
//        t8 = *t7
//        t9 = t8()
//        t10 = &t3.foo [#0]                                              *func()
//        t11 = *t10                                                       func()
//        t12 = &t3.do [#1]                                                 *Doer
//        t13 = changetype Doer <- func() (t11)                              Doer
//        *t12 = t13
//        t14 = &t3.do [#1]                                                 *Doer
//        t15 = *t14                                                         Doer
//        t16 = t15()                                                          ()

// Flow chain showing that Baz$1 reaches t8():
//   Baz$1 -> t2 <-> PtrFunction(func()) <-> t4 -> t5 -> t6 <-> Field(testdata.A:foo) <-> t7 -> t8
// Flow chain showing that Baz$1 reaches t15():
//  Field(testdata.A:foo) <-> t10 -> t11 -> t13 -> t12 <-> Field(testdata.A:do) <-> t14 -> t15

// WANT:
// Local(f) -> Local(t0)
// Local(t0) -> PtrFunction(func())
// Function(Baz$1) -> Local(t2)
// PtrFunction(func()) -> Local(t0), Local(t2), Local(t4)
// Local(t2) -> PtrFunction(func())
// Local(t6) -> Field(testdata.A:foo)
// Local(t4) -> Local(t5), PtrFunction(func())
// Local(t5) -> Local(t6)
// Local(t7) -> Field(testdata.A:foo), Local(t8)
// Field(testdata.A:foo) -> Local(t10), Local(t6), Local(t7)
// Local(t6) -> Field(testdata.A:foo)
// Field(testdata.A:do) -> Local(t12), Local(t14)
// Local(t12) -> Field(testdata.A:do)
// Local(t10) -> Field(testdata.A:foo), Local(t11)
// Local(t11) -> Local(t13)
// Local(t13) -> Local(t12)
// Local(t14) -> Field(testdata.A:do), Local(t15)
