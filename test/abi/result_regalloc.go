// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bug: in (*bb).d, the value to be returned was not allocated to
// a register that satisfies its register mask.

package main

type bb struct {
	r float64
	x []float64
}

//go:noinline
func B(r float64, x []float64) I {
	return bb{r, x}
}

func (b bb) d() (int, int) {
	if b.r == 0 {
		return 0, len(b.x)
	}
	return len(b.x), len(b.x)
}

type I interface { d() (int, int) }

func D(r I) (int, int) { return r.d() }

//go:noinline
func F() (int, int) {
	r := float64(1)
	x := []float64{0, 1, 2}
	b := B(r, x)
	return D(b)
}

func main() {
	x, y := F()
	if x != 3 || y != 3 {
		panic("FAIL")
	}
}
