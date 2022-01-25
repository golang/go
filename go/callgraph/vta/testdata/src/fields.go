// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type J interface {
	I
	Bar()
}

type A struct{}

func (a A) Foo() {}
func (a A) Bar() {}

type B struct {
	a A
	i I
}

func Do() B {
	b := B{}
	return b
}

func Baz(b B) {
	var j J
	j = b.a

	j.Bar()

	b.i = j

	Do().i.Foo()
}

// Relevant SSA:
// func Baz(b B):
//   t0 = local B (b)
//   *t0 = b
//   t1 = &t0.a [#0]       // no flow here since a is of concrete type
//   t2 = *t1
//   t3 = make J <- A (t2)
//   t4 = invoke t3.Bar()
//   t5 = &t0.i [#1]
//   t6 = change interface I <- J (t3)
//   *t5 = t6
//   t7 = Do()
//   t8 = t7.i [#0]
//   t9 = (A).Foo(t8)
//   return

// WANT:
// Field(testdata.B:i) -> Local(t5), Local(t8)
// Local(t5) -> Field(testdata.B:i)
// Local(t2) -> Local(t3)
// Local(t3) -> Local(t6)
// Local(t6) -> Local(t5)
