// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type J interface {
	Foo()
	Bar()
}

type B struct {
	p int
}

func (b B) Foo() {}
func (b B) Bar() {}

func Baz(b *B, S []*I, s []J) {
	var x [3]I
	x[1] = b

	a := &s[2]
	(*a).Bar()

	print([3]*I{nil, nil, nil}[2])
}

// Relevant SSA:
// func Baz(b *B, S []*I, s []J):
//   t0 = local [3]I (x)
//   t1 = &t0[1:int]
//   ...
//   t3 = &s[2:int]
//   t4 = *t3
//   ...
//   t6 = local [3]*I (complit)
//   t7 = &t6[0:int]
//         ...
//   t11 = t10[2:int]
//   ...

// WANT:
// Slice([]testdata.I) -> Local(t1)
// Local(t1) -> Slice([]testdata.I)
// Slice([]testdata.J) -> Local(t3)
// Local(t3) -> Local(t4), Slice([]testdata.J)
// Local(t11) -> Slice([]*testdata.I)
// Slice([]*testdata.I) -> Local(t11), PtrInterface(testdata.I)
// Constant(*testdata.I) -> PtrInterface(testdata.I)
