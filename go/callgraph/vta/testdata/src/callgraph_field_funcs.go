// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type WrappedFunc struct {
	F func() complex64
}

func callWrappedFunc(f WrappedFunc) {
	f.F()
}

func foo() complex64 {
	println("foo")
	return -1
}

func Foo(b bool) {
	callWrappedFunc(WrappedFunc{foo})
	x := func() {}
	y := func() {}
	var a *func()
	if b {
		a = &x
	} else {
		a = &y
	}
	(*a)()
}

// Relevant SSA:
// func Foo(b bool):
//         t0 = local WrappedFunc (complit)
//         t1 = &t0.F [#0]
//         *t1 = foo
//         t2 = *t0
//         t3 = callWrappedFunc(t2)
//         t4 = new func() (x)
//         *t4 = Foo$1
//         t5 = new func() (y)
//         *t5 = Foo$2
//         if b goto 1 else 3
// 1:
//         jump 2
// 2:
//         t6 = phi [1: t4, 3: t5] #a
//         t7 = *t6
//         t8 = t7()
//         return
// 3:
//         jump 2
//
// func callWrappedFunc(f WrappedFunc):
//         t0 = local WrappedFunc (f)
//         *t0 = f
//         t1 = &t0.F [#0]
//         t2 = *t1
//         t3 = t2()
//         return

// WANT:
// callWrappedFunc: t2() -> foo
// Foo: callWrappedFunc(t2) -> callWrappedFunc; t7() -> Foo$1, Foo$2
