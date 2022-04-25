// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

func instantiated[X any](x *X) int {
	print(x)
	return 0
}

type I interface {
	Bar()
}

func interfaceInstantiated[X I](x X) {
	x.Bar()
}

type A struct{}

func (a A) Bar() {}

type B struct{}

func (b B) Bar() {}

func Foo(a A, b B) {
	x := true
	instantiated[bool](&x)
	y := 1
	instantiated[int](&y)

	interfaceInstantiated[A](a)
	interfaceInstantiated[B](b)
}

// Relevant SSA:
//func Foo(a A, b B):
//  t0 = local A (a)
//  *t0 = a
//  t1 = local B (b)
//  *t1 = b
//  t2 = new bool (x)
//  *t2 = true:bool
//  t3 = instantiated[[bool]](t2)
//  t4 = new int (y)
//  *t4 = 1:int
//  t5 = instantiated[[int]](t4)
//  t6 = *t0
//  t7 = interfaceInstantiated[[testdata.A]](t6)
//  t8 = *t1
//  t9 = interfaceInstantiated[[testdata.B]](t8)
//  return
//
//func interfaceInstantiated[[testdata.B]](x B):
//  t0 = local B (x)
//  *t0 = x
//  t1 = *t0
//  t2 = (B).Bar(t1)
//  return
//
//func interfaceInstantiated[X I](x X):
//        (external)

// WANT:
// Foo: instantiated[[bool]](t2) -> instantiated[[bool]]; instantiated[[int]](t4) -> instantiated[[int]]; interfaceInstantiated[[testdata.A]](t6) -> interfaceInstantiated[[testdata.A]]; interfaceInstantiated[[testdata.B]](t8) -> interfaceInstantiated[[testdata.B]]
// interfaceInstantiated[[testdata.B]]: (B).Bar(t1) -> B.Bar
// interfaceInstantiated[[testdata.A]]: (A).Bar(t1) -> A.Bar
