// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S int

type T struct {
	a int
	S
}

//go:noinline
func (s *S) M(a int, x [2]int, b float64, y [2]float64) (S, int, [2]int, float64, [2]float64) {
	return *s, a, x, b, y
}

var s S = 42
var t = &T{S: s}

var fn = (*T).M // force a method wrapper

func main() {
	a := 123
	x := [2]int{456, 789}
	b := 1.2
	y := [2]float64{3.4, 5.6}
	s1, a1, x1, b1, y1 := fn(t, a, x, b, y)
	if a1 != a || x1 != x || b1 != b || y1 != y || s1 != s {
		panic("FAIL")
	}
}
