// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type B struct {
	p int
}

func (b B) Foo() {}

func Baz(i, j *I, b, c bool) {
	if b {
		i = j
	}
	*i = B{9}
	if c {
		(*i).Foo()
	} else {
		(*j).Foo()
	}
}

// Relevant SSA:
// func Baz(i *I, j *I, b bool, c bool):
//    if b goto 1 else 2
//  1:
//    jump 2
//  2:
//    t0 = phi [0: i, 1: j] #i
//    t1 = local B (complit)
//    t2 = &t1.p [#0]
//    *t2 = 9:int
//    t3 = *t1
//    t4 = make I <- B (t3)
//    *t0 = t4
//    if c goto 3 else 5
//  3:
//    t5 = *t0
//    t6 = invoke t5.Foo()
//    jump 4
//  4:
//    return
//  5:
//    t7 = *j
//    t8 = invoke t7.Foo()
//    jump 4

// Flow chain showing that B reaches (*i).foo():
//   t3 (B) -> t4 -> t0 -> t5
// Flow chain showing that B reaches (*j).foo():
//   t3 (B) -> t4 -> t0 <--> j -> t7

// WANT:
// Local(t0) -> Local(i), Local(j), Local(t5)
// Local(i) -> Local(t0)
// Local(j) -> Local(t0), Local(t7)
// Local(t3) -> Local(t4)
// Local(t4) -> Local(t0)
