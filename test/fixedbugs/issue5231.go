// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5231: method values lose their variadic property.

package p

type T int

func (t T) NotVariadic(s []int) int {
	return int(t) + s[0]
}

func (t T) Variadic(s ...int) int {
	return int(t) + s[0]
}

type I interface {
	NotVariadic(s []int) int
	Variadic(s ...int) int
}

func F() {
	var t T
	var p *T = &t
	var i I = p

	nv := t.NotVariadic
	nv = p.NotVariadic
	nv = i.NotVariadic
	var s int = nv([]int{1, 2, 3})

	v := t.Variadic
	v = p.Variadic
	v = i.Variadic
	s = v(1, 2, 3)

	var f1 func([]int) int = nv
	var f2 func(...int) int = v

	_, _, _ = f1, f2, s
}
