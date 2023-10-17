// compile -d=ssa/check/on

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var i int

type t struct {
	a, b, c, d, e int
}

func f(p t, q int) int {
	var a, b, c, d, e, f, g int
	var h, i, j, k, l, m int
	_, _, _, _, _, _, _ = a, b, c, d, e, f, g
	_, _, _, _, _, _ = h, i, j, k, l, m
	return 0
}

func g() int {
	var v t
	return f(v, 1<<i)
}
