// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo() string
}

type J interface {
	Foo() string
	Bar()
}

type B struct {
	p string
}

func (b B) Foo() string { return b.p }
func (b B) Bar()        {}

func Baz(m map[I]I, b1, b2 B, n map[string]*J) *J {
	m[b1] = b2

	return n[b1.Foo()]
}

// Relevant SSA:
// func Baz(m map[I]I, b1 B, b2 B, n map[string]*J) *J:
//   t0 = local B (b1)
//   *t0 = b1
//   t1 = local B (b2)
//   *t1 = b2
//   t2 = *t0
//   t3 = make I <- B (t2)
//   t4 = *t1
//   t5 = make I <- B (t4)
//   m[t3] = t5
//   t6 = *t0
//   t7 = (B).Foo(t6)
//   t8 = n[t7]
//   return t8

// WANT:
// Local(t4) -> Local(t5)
// Local(t5) -> MapValue(testdata.I)
// Local(t3) -> MapKey(testdata.I)
// Local(t8) -> MapValue(*testdata.J)
// MapValue(*testdata.J) -> Local(t8)
