// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

// TestNodeTypeUniqueness checks if semantically equivalent types
// are being represented using the same pointer value in vta nodes.
// If not, some edges become missing in the string representation
// of the graph.

type I interface {
	Foo()
}

type A struct{}

func (a A) Foo() {}

func Baz(a *A) (I, I, interface{}, interface{}) {
	var i I
	i = a

	var ii I
	aa := &A{}
	ii = aa

	m := make(map[int]int)
	var iii interface{}
	iii = m

	var iiii interface{}
	iiii = m

	return i, ii, iii, iiii
}

// Relevant SSA:
// func Baz(a *A) (I, I, interface{}, interface{}):
//   t0 = make I <- *A (a)
//	 t1 = new A (complit)
//   t2 = make I <- *A (t1)
//   t3 = make map[int]int
//   t4 = make interface{} <- map[int]int (t3)
//   t5 = make interface{} <- map[int]int (t3)
//   return t0, t2, t4, t5

// Without canon approach, one of Pointer(*A) -> Local(t0) and Pointer(*A) -> Local(t2) edges is
// missing in the graph string representation. The original graph has both of the edges but the
// source node Pointer(*A) is not the same; two occurences of Pointer(*A) are considered separate
// nodes. Since they have the same string representation, one edge gets overriden by the other
// during the graph stringification, instead of being joined together as in below.

// WANT:
// Pointer(*testdata.A) -> Local(t0), Local(t2)
// Local(t3) -> Local(t4), Local(t5)
