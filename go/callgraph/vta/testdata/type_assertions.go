// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

// Test program for testing type assertions and extract instructions.
// The latter are tested here too since extract instruction comes
// naturally in type assertions.

type I interface {
	Foo()
}

type J interface {
	Foo()
	Bar()
}

type A struct {
	c int
}

func (a A) Foo() {}
func (a A) Bar() {}

func Baz(i I) {
	j, ok := i.(J)
	if ok {
		j.Foo()
	}

	a := i.(*A)
	a.Bar()
}

// Relevant SSA:
// 	func Baz(i I):
//    t0 = typeassert,ok i.(J)
//    t1 = extract t0 #0
//    t2 = extract t0 #1
//    if t2 goto 1 else 2
//  1:
//    t3 = invoke t1.Foo()
//    jump 2
//  2:
//    t4 = typeassert i.(*A)  // no flow since t4 is of concrete type
//    t5 = *t4
//    t6 = (A).Bar(t5)
//    return

// WANT:
// Local(i) -> Local(t0[0])
// Local(t0[0]) -> Local(t1)
