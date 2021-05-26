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
//  t0 = local B (b)
//  *t0 = b
//  if c goto 1 else 3
//
// 1:
//  t1 = *t0
//  t2 = make I <- B (t1)
//  jump 2
//
// 2:
//  t3 = phi [1: t2, 3: t7] #i
//  t4 = invoke t3.foo()
//  return
//
// 3:
//  t5 = local A (a)
//  t6 = *t5
//  t7 = make I <- A (t6)
//  jump 2

// WANT:
// Local(t1) -> Local(t2)
// Local(t2) -> Local(t3)
// Local(t7) -> Local(t3)
// Local(t6) -> Local(t7)
