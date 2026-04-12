// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

const arrFooSize = 96

type arrFoo [arrFooSize]int

//go:noinline
func badCopy(dst, src []int) {
	p := (*[arrFooSize]int)(dst[:arrFooSize])
	q := (*[arrFooSize]int)(src[:arrFooSize])
	*p = arrFoo(*q)
}

//go:noinline
func goodCopy(dst, src []int) {
	p := (*[arrFooSize]int)(dst[:arrFooSize])
	q := (*[arrFooSize]int)(src[:arrFooSize])
	*p = *q
}

func TestOverlapedMoveWithNoopIConv(t *testing.T) {
	h1 := make([]int, arrFooSize+1)
	h2 := make([]int, arrFooSize+1)
	for i := range arrFooSize + 1 {
		h1[i] = i
		h2[i] = i
	}
	badCopy(h1[1:], h1[:arrFooSize])
	goodCopy(h2[1:], h2[:arrFooSize])
	for i := range arrFooSize + 1 {
		if h1[i] != h2[i] {
			t.Errorf("h1 and h2 differ at index %d, expect them to be the same", i)
		}
	}
}
