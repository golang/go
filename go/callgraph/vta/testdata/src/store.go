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
//	t0 = new I (i)
//	t1 = make I <- A (struct{}{}:A)    A  -> t1
//	*t0 = t1                           t1 -> t0
//	t2 = *t0                           t0 -> t2
//	t3 = invoke t2.foo()
//	return

// WANT:
// Constant(testdata.A) -> Local(t1)
// Local(t1) -> Local(t0)
// Local(t0) -> Local(t2)
