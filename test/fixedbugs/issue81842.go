// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 81842: on ppc64 and ppc64le, signed division compared the
// divisor with -1 in CR0 without marking the op as clobbering flags, so
// flagalloc could keep a flag live across the division and a branch
// after it tested the wrong condition.

package main

import (
	"fmt"
	"math"
)

var sinkA, sinkB int
var q64 int64
var q32 int32

//go:noinline
func div64(c bool, a, b int64) {
	sinkB = 0
	if b == 0 { // b is known nonzero below, so the division has no zero check
		return
	}
	if c {
		sinkA = 1
	}
	q64 = a / b
	if !c {
		sinkB = 1
	}
}

//go:noinline
func div32(c bool, a, b int32) {
	sinkB = 0
	if b == 0 {
		return
	}
	if c {
		sinkA = 1
	}
	q32 = a / b
	if !c {
		sinkB = 1
	}
}

//go:noinline
func div16(c bool, a, b int16) {
	sinkB = 0
	if b == 0 {
		return
	}
	if c {
		sinkA = 1
	}
	q64 = int64(a / b)
	if !c {
		sinkB = 1
	}
}

//go:noinline
func div8(c bool, a, b int8) {
	sinkB = 0
	if b == 0 {
		return
	}
	if c {
		sinkA = 1
	}
	q64 = int64(a / b)
	if !c {
		sinkB = 1
	}
}

//go:noinline
func mod64(c bool, a, b int64) {
	sinkB = 0
	if b == 0 {
		return
	}
	if c {
		sinkA = 1
	}
	q64 = a % b
	if !c {
		sinkB = 1
	}
}

//go:noinline
func divCmp(x int, a, b int64) {
	sinkB = 0
	if b == 0 {
		return
	}
	if x < 5 {
		sinkA = 1
	}
	q64 = a / b
	if x >= 5 {
		sinkB = 1
	}
}

func check(name string, gotQ, wantQ int64, gotB, wantB int) {
	if gotQ != wantQ || gotB != wantB {
		panic(fmt.Sprintf("%s: got q=%d sinkB=%d, want q=%d sinkB=%d", name, gotQ, gotB, wantQ, wantB))
	}
}

func main() {
	div64(false, 7, 2)
	check("div64", q64, 3, sinkB, 1)
	div64(false, math.MinInt64, -1)
	check("div64(MinInt64, -1)", q64, math.MinInt64, sinkB, 1)
	div32(false, 7, 2)
	check("div32", int64(q32), 3, sinkB, 1)
	div32(false, math.MinInt32, -1)
	check("div32(MinInt32, -1)", int64(q32), math.MinInt32, sinkB, 1)
	div16(false, 7, 2)
	check("div16", q64, 3, sinkB, 1)
	div8(false, 7, 2)
	check("div8", q64, 3, sinkB, 1)
	mod64(false, 7, 2)
	check("mod64", q64, 1, sinkB, 1)
	divCmp(0, 7, 2)
	check("divCmp(0)", q64, 3, sinkB, 0)
	divCmp(9, 7, -2)
	check("divCmp(9)", q64, -3, sinkB, 1)
}
