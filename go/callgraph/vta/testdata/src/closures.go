// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

func Do(i I) { i.Foo() }

func Baz(b bool, h func(I)) {
	var i I
	a := func(g func(I)) {
		g(i)
	}

	if b {
		h = Do
	}

	a(h)
}

// Relevant SSA:
//  func Baz(b bool, h func(I)):
//    t0 = new I (i)
//    t1 = make closure Baz$1 [t0]
//    if b goto 1 else 2
//   1:
//         jump 2
//   2:
//    t2 = phi [0: h, 1: Do] #h
//    t3 = t1(t2)
//    return
//
// func Baz$1(g func(I)):
//    t0 = *i
//    t1 = g(t0)
//    return

// In the edge set Local(i) -> Local(t0), Local(t0) below,
// two occurrences of t0 come from t0 in Baz and Baz$1.

// WANT:
// Function(Do) -> Local(t2)
// Function(Baz$1) -> Local(t1)
// Local(h) -> Local(t2)
// Local(t0) -> Local(i)
// Local(i) -> Local(t0), Local(t0)
