// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	foo()
}

type A struct{}

func (a A) foo() {}

func recover1() {
	print("only this recover should execute")
	if r, ok := recover().(I); ok {
		r.foo()
	}
}

func recover2() {
	recover()
}

func Baz(a A) {
	defer recover1()
	defer recover()
	panic(a)
}

// Relevant SSA:
// func recover1():
//   t0 = print("only this recover...":string)
//   t1 = recover()
//   t2 = typeassert,ok t1.(I)
//   t3 = extract t2 #0
//   t4 = extract t2 #1
//   if t4 goto 1 else 2
//  1:
//   t5 = invoke t3.foo()
//   jump 2
//  2:
//   return
//
// func recover2():
//   t0 = recover()
//   return
//
// func Baz(i I):
//   defer recover1()
//   t0 = make interface{} <- A (a)
//   panic t2

// t0 argument to panic in Baz gets ultimately connected to recover
// registers t1 in recover1() and t0 in recover2().

// WANT:
// Panic -> Recover
// Local(t0) -> Panic
// Recover -> Local(t0), Local(t1)
