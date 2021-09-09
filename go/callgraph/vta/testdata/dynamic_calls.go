// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	foo(I)
}

type A struct{}

func (a A) foo(ai I) {}

type B struct{}

func (b B) foo(bi I) {}

func doWork() I { return nil }
func close() I  { return nil }

func Baz(x B, h func() I, i I) I {
	i.foo(x)

	return h()
}

// Relevant SSA:
// func Baz(x B, h func() I, i I) I:
//   t0 = local B (x)
//   *t0 = x
//   t1 = *t0
//   t2 = make I <- B (t1)
//   t3 = invoke i.foo(t2)
//   t4 = h()
//   return t4

// Local(t2) has seemingly duplicates of successors. This
// happens in stringification of type propagation graph.
// Due to CHA, we analyze A.foo and *A.foo as well as B.foo
// and *B.foo, which have similar bodies and hence similar
// type flow that gets merged together during stringification.

// WANT:
// Local(t2) -> Local(ai), Local(ai), Local(bi), Local(bi)
// Constant(testdata.I) -> Local(t4)
// Local(t1) -> Local(t2)
