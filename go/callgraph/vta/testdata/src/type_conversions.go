// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type Y interface {
	Foo()
	Bar(float64)
}

type Z Y

type W interface {
	Y
}

type A struct{}

func (a A) Foo()          { print("A:Foo") }
func (a A) Bar(f float64) { print(uint(f)) }

type B struct{}

func (b B) Foo()          { print("B:Foo") }
func (b B) Bar(f float64) { print(uint(f) + 1) }

type X interface {
	Foo()
}

func Baz(y Y) {
	z := Z(y)
	z.Foo()

	x := X(y)
	x.Foo()

	y = A{}
	var y_p *Y = &y

	w_p := (*W)(y_p)
	*w_p = B{}

	(*y_p).Foo() // prints B:Foo
	(*w_p).Foo() // prints B:Foo
}

// Relevant SSA:
//  func Baz(y Y):
//   t0 = new Y (y)
//   *t0 = y
//   t1 = *t0
//   t2 = changetype Z <- Y (t1)
//   t3 = invoke t2.Foo()
//
//   t4 = *t0
//   t5 = change interface X <- Y (t4)
//   t6 = invoke t5.Foo()
//
//   t7 = make Y <- A (struct{}{}:A)
//   *t0 = t7
//   t8 = changetype *W <- *Y (t0)
//   t9 = make W <- B (struct{}{}:B)
//   *t8 = t9
//   t10 = *t0
//   t11 = invoke t10.Foo()
//   t12 = *t8
//   t13 = invoke t12.Foo()
//   return

// WANT:
// Local(t1) -> Local(t2)
// Local(t4) -> Local(t5)
// Local(t0) -> Local(t1), Local(t10), Local(t4), Local(t8)
// Local(y) -> Local(t0)
// Constant(testdata.A) -> Local(t7)
// Local(t7) -> Local(t0)
// Local(t9) -> Local(t8)
