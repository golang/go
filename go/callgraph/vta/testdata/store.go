// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

// Tests graph creation for store/load and make instructions.
// Note that ssa package does not have a load instruction per
// se. Yet, one is encoded as a unary instruction with the
// * operator.

type A struct{}

type I interface{ foo() }

func (a A) foo() {}

func main() {
	a := A{}
	var i I
	i = a
	ii := &i
	(*ii).foo()
}

// Relevant SSA:
//	t0 = local A (a)
//	t1 = new I (i)
//	t2 = *t0                 no interesting flow: concrete types
//	t3 = make I <- A (t2)    t2 -> t3
//	*t1 = t3                 t3 -> t1
//	t4 = *t1                 t1 -> t4
//	t5 = invoke t4.foo()
//	return

// WANT:
// Local(t2) -> Local(t3)
// Local(t3) -> Local(t1)
// Local(t1) -> Local(t4)
