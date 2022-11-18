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
//  t0 = new bool (x)
//  *t0 = true:bool
//  t1 = instantiated[bool](t2)
//  t1 = new int (y)
//  *t2 = 1:int
//  t3 = instantiated[[int]](t4)
//  t4 = interfaceInstantiated[testdata.A](a)
//  t5 = interfaceInstantiated[testdata.B](b)
//  return
//
//func interfaceInstantiated[[testdata.B]](x B):
//  t0 = (B).Bar(b)
//  return
//
//func interfaceInstantiated[X I](x X):
//        (external)

// WANT:
// Foo: instantiated[bool](t0) -> instantiated[bool]; instantiated[int](t2) -> instantiated[int]; interfaceInstantiated[testdata.A](a) -> interfaceInstantiated[testdata.A]; interfaceInstantiated[testdata.B](b) -> interfaceInstantiated[testdata.B]
// interfaceInstantiated[testdata.B]: (B).Bar(x) -> B.Bar
// interfaceInstantiated[testdata.A]: (A).Bar(x) -> A.Bar
