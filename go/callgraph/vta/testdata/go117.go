// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type J interface {
	Foo()
	Bar()
}

type B struct {
	p int
}

func (b B) Foo() {}
func (b B) Bar() {}

func Wobble(b *B, s []J) {
	x := (*[3]J)(s)
	x[1] = b

	a := &s[2]
	(*a).Bar()
}

// Relevant SSA:
// func Wobble(b *B, s []J):
//   t0 = slice to array pointer *[3]J <- []J (s)                      *[3]J
//   t1 = &t0[1:int]                                                      *J
//   t2 = make J <- *B (b)                                                 J
//   *t1 = t2
//   t3 = &s[2:int]                                                       *J
//   ...

// WANT:
// Local(t1) -> Slice([]testdata.J)
// Slice([]testdata.J) -> Local(t1), Local(t3)
