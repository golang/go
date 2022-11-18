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
//   t0 = make I <- B (b1)
//   t1 = make I <- B (b2)
//   m[t0] = t1
//   t2 = (B).Foo(b1)
//   t3 = n[t2]
//   return t3

// WANT:
// Local(b2) -> Local(t1)
// Local(t1) -> MapValue(testdata.I)
// Local(t0) -> MapKey(testdata.I)
// Local(t3) -> MapValue(*testdata.J)
// MapValue(*testdata.J) -> Local(t3)
