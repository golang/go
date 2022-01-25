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

func Do(a A, b B) map[I]I {
	m := make(map[I]I)
	m[a] = B{}
	m[b] = b
	return m
}

func Baz(a A, b B) {
	var x []I
	for k, v := range Do(a, b) {
		k.Foo()
		v.Foo()

		x = append(x, k)
	}

	x[len(x)-1].Foo()
}

// Relevant SSA:
// func Baz(a A, b B):
//   ...
//   t4 = Do(t2, t3)
//   t5 = range t4
//   jump 1
//  1:
//   t6 = phi [0: nil:[]I, 2: t16] #x
//   t7 = next t5
//   t8 = extract t7 #0
//   if t8 goto 2 else 3
//  2:
//   t9 = extract t7 #1
//   t10 = extract t7 #2
//   t11 = invoke t9.Foo()
//   t12 = invoke t10.Foo()
//   ...
//   jump 1
//  3:
//   t17 = len(t6)
//   t18 = t17 - 1:int
//   t19 = &t6[t18]
//   t20 = *t19
//   t21 = invoke t20.Foo()
//   return

// WANT:
// Baz: Do(t2, t3) -> Do; invoke t10.Foo() -> B.Foo; invoke t20.Foo() -> A.Foo, B.Foo; invoke t9.Foo() -> A.Foo, B.Foo
