// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// splitload rewrites must preserve pointer-typed loads so
// spilled values remain visible to stack maps across stack growth.

package main

type splitLoadObj struct{ x int }

var sink byte

//go:noinline
func splitLoadGrow(n int) {
	var buf [2048]byte
	buf[n&2047] = byte(n)
	if n > 0 {
		splitLoadGrow(n - 1)
	}
	sink = buf[n&2047]
}

//go:noinline
func splitLoadPointerCompare(pp **splitLoadObj, q *splitLoadObj, a, b, m int) int {
	cond := q == *pp
	x := a
	if cond {
		x = b
	}
	z := x & m
	splitLoadGrow(1000)
	if cond {
		return z
	}
	return x
}

func main() {
	var obj splitLoadObj
	slot := &obj
	if got := splitLoadPointerCompare(&slot, &obj, 10, 6, 1); got != 0 {
		println("splitLoadPointerCompare(...) =", got, "want 0")
		panic("FAIL")
	}
}
