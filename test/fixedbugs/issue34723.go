// errorcheck -0 -d=wb

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't introduce write barriers where we
// don't need them. These cases are writing pointers to
// globals to zeroed memory.

package main

func f1() []string {
	return []string{"a"}
}

func f2() []string {
	return []string{"a", "b"}
}

type T struct {
	a [6]*int
}

func f3() *T {
	t := new(T)
	t.a[0] = &g
	t.a[1] = &g
	t.a[2] = &g
	t.a[3] = &g
	t.a[4] = &g
	t.a[5] = &g
	return t
}

func f4() *T {
	t := new(T)
	t.a[5] = &g
	t.a[4] = &g
	t.a[3] = &g
	t.a[2] = &g
	t.a[1] = &g
	t.a[0] = &g
	return t
}

func f5() *T {
	t := new(T)
	t.a[4] = &g
	t.a[2] = &g
	t.a[0] = &g
	t.a[3] = &g
	t.a[1] = &g
	t.a[5] = &g
	return t
}

type U struct {
	a [65]*int
}

func f6() *U {
	u := new(U)
	u.a[63] = &g
	// This offset is too large: we only track the first 64 pointers for zeroness.
	u.a[64] = &g // ERROR "write barrier"
	return u
}

var g int
