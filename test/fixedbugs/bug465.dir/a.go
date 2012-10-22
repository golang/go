// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct{ A, B int }

type A []int

type M map[int]int

func F1() int {
	if (T{1, 2}) == (T{3, 4}) {
		return 1
	}
	return 0
}

func F2() int {
	if (M{1: 2}) == nil {
		return 1
	}
	return 0
}

func F3() int {
	if nil == (A{}) {
		return 1
	}
	return 0
}

func F4() int {
	if a := (A{}); a == nil {
		return 1
	}
	return 0
}

func F5() int {
	for k, v := range (M{1: 2}) {
		return v - k
	}
	return 0
}

func F6() int {
	switch a := (T{1, 1}); a == (T{1, 2}) {
	default:
		return 1
	}
	return 0
}

func F7() int {
	for m := (M{}); len(m) < (T{1, 2}).A; m[1] = (A{1})[0] {
		return 1
	}
	return 0
}
