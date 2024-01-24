// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 61992, inconsistent 'mem' juggling in expandCalls

package p

type S1 struct {
	a, b, c []int
	i       int
}

type S2 struct {
	a, b []int
	m    map[int]int
}

func F(i int, f func(S1, S2, int) int) int {
	return f(
		S1{},
		S2{m: map[int]int{}},
		1<<i)
}
