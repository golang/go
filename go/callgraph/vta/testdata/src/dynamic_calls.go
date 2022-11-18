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

var g *B = &B{} // ensure *B.foo is created.

// Relevant SSA:
// func Baz(x B, h func() I, i I) I:
//   t0 = make I <- B (x)
//   t1 = invoke i.foo(t0)
//   t2 = h()
//   return t2

// Local(t0) has seemingly duplicates of successors. This
// happens in stringification of type propagation graph.
// Due to CHA, we analyze A.foo and *A.foo as well as B.foo
// and *B.foo, which have similar bodies and hence similar
// type flow that gets merged together during stringification.

// WANT:
// Local(t0) -> Local(ai), Local(ai), Local(bi), Local(bi)
// Constant(testdata.I) -> Local(t2)
// Local(x) -> Local(t0)
