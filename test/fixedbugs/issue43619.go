// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

//go:noinline
func fcmplt(a, b float64, x uint64) uint64 {
	if a < b {
		x = 0
	}
	return x
}

//go:noinline
func fcmple(a, b float64, x uint64) uint64 {
	if a <= b {
		x = 0
	}
	return x
}

//go:noinline
func fcmpgt(a, b float64, x uint64) uint64 {
	if a > b {
		x = 0
	}
	return x
}

//go:noinline
func fcmpge(a, b float64, x uint64) uint64 {
	if a >= b {
		x = 0
	}
	return x
}

//go:noinline
func fcmpeq(a, b float64, x uint64) uint64 {
	if a == b {
		x = 0
	}
	return x
}

//go:noinline
func fcmpne(a, b float64, x uint64) uint64 {
	if a != b {
		x = 0
	}
	return x
}

func main() {
	type fn func(a, b float64, x uint64) uint64

	type testCase struct {
		f       fn
		a, b    float64
		x, want uint64
	}
	NaN := math.NaN()
	for _, t := range []testCase{
		{fcmplt, 1.0, 1.0, 123, 123},
		{fcmple, 1.0, 1.0, 123, 0},
		{fcmpgt, 1.0, 1.0, 123, 123},
		{fcmpge, 1.0, 1.0, 123, 0},
		{fcmpeq, 1.0, 1.0, 123, 0},
		{fcmpne, 1.0, 1.0, 123, 123},

		{fcmplt, 1.0, 2.0, 123, 0},
		{fcmple, 1.0, 2.0, 123, 0},
		{fcmpgt, 1.0, 2.0, 123, 123},
		{fcmpge, 1.0, 2.0, 123, 123},
		{fcmpeq, 1.0, 2.0, 123, 123},
		{fcmpne, 1.0, 2.0, 123, 0},

		{fcmplt, 2.0, 1.0, 123, 123},
		{fcmple, 2.0, 1.0, 123, 123},
		{fcmpgt, 2.0, 1.0, 123, 0},
		{fcmpge, 2.0, 1.0, 123, 0},
		{fcmpeq, 2.0, 1.0, 123, 123},
		{fcmpne, 2.0, 1.0, 123, 0},

		{fcmplt, 1.0, NaN, 123, 123},
		{fcmple, 1.0, NaN, 123, 123},
		{fcmpgt, 1.0, NaN, 123, 123},
		{fcmpge, 1.0, NaN, 123, 123},
		{fcmpeq, 1.0, NaN, 123, 123},
		{fcmpne, 1.0, NaN, 123, 0},

		{fcmplt, NaN, 1.0, 123, 123},
		{fcmple, NaN, 1.0, 123, 123},
		{fcmpgt, NaN, 1.0, 123, 123},
		{fcmpge, NaN, 1.0, 123, 123},
		{fcmpeq, NaN, 1.0, 123, 123},
		{fcmpne, NaN, 1.0, 123, 0},

		{fcmplt, NaN, NaN, 123, 123},
		{fcmple, NaN, NaN, 123, 123},
		{fcmpgt, NaN, NaN, 123, 123},
		{fcmpge, NaN, NaN, 123, 123},
		{fcmpeq, NaN, NaN, 123, 123},
		{fcmpne, NaN, NaN, 123, 0},
	} {
		got := t.f(t.a, t.b, t.x)
		if got != t.want {
			panic(fmt.Sprintf("want %v, got %v", t.want, got))
		}
	}
}
