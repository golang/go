// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type A struct{}
type B struct{}

type I interface{ foo() }

func (a A) foo() {}
func (b B) foo() {}

func Baz(b B, c bool) {
	var i I
	if c {
		i = b
	} else {
		a := A{}
		i = a
	}
	i.foo()
}

// Relevant SSA:
// func Baz(b B, c bool):
// 0:
//  if c goto 1 else 3
//
// 1:
//  t0 = make I <- B (b)
//  jump 2
//
// 2:
//  t1 = phi [1: t0, 3: t3] #i
//  t2 = invoke t1.foo()
//  return
//
// 3:
//  t3 = make I <- A (struct{}{}:A)
//  jump 2

// WANT:
// Local(b) -> Local(t0)
// Local(t0) -> Local(t1)
// Local(t3) -> Local(t1)
// Constant(testdata.A) -> Local(t3)
