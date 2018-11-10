// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7044: bad AMOVFD and AMOVDF assembly generation on
// arm for registers above 7.

package main

import (
	"fmt"
	"reflect"
)

func f() [16]float32 {
	f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15 :=
		float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)
	// Use all 16 registers to do float32 --> float64 conversion.
	d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15 :=
		float64(f0), float64(f1), float64(f2), float64(f3), float64(f4), float64(f5), float64(f6), float64(f7), float64(f8), float64(f9), float64(f10), float64(f11), float64(f12), float64(f13), float64(f14), float64(f15)
	// Use all 16 registers to do float64 --> float32 conversion.
	g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15 :=
		float32(d0), float32(d1), float32(d2), float32(d3), float32(d4), float32(d5), float32(d6), float32(d7), float32(d8), float32(d9), float32(d10), float32(d11), float32(d12), float32(d13), float32(d14), float32(d15)
	// Force another conversion, so that the previous conversion doesn't
	// get optimized away into constructing the returned array. With current
	// optimizations, constructing the returned array uses only
	// a single register.
	e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 :=
		float64(g0), float64(g1), float64(g2), float64(g3), float64(g4), float64(g5), float64(g6), float64(g7), float64(g8), float64(g9), float64(g10), float64(g11), float64(g12), float64(g13), float64(g14), float64(g15)
	return [16]float32{
		float32(e0), float32(e1), float32(e2), float32(e3), float32(e4), float32(e5), float32(e6), float32(e7), float32(e8), float32(e9), float32(e10), float32(e11), float32(e12), float32(e13), float32(e14), float32(e15),
	}
}

func main() {
	want := [16]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	got := f()
	if !reflect.DeepEqual(got, want) {
		fmt.Printf("f() = %#v; want %#v\n", got, want)
	}
}
