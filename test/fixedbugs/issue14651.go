// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test checks if the compiler's internal constant
// arithmetic correctly rounds up floating-point values
// that become the smallest denormal value.
//
// See also related issue 14553 and test issue14553.go.

package main

import (
	"fmt"
	"math"
)

const (
	p149 = 1.0 / (1 << 149) // 1p-149
	p500 = 1.0 / (1 << 500) // 1p-500
	p1074 = p500 * p500 / (1<<74) // 1p-1074
)

const (
	m0000p149 = 0x0 / 16.0 * p149 // = 0.0000p-149
	m1000p149 = 0x8 / 16.0 * p149 // = 0.1000p-149
	m1001p149 = 0x9 / 16.0 * p149 // = 0.1001p-149
	m1011p149 = 0xb / 16.0 * p149 // = 0.1011p-149
	m1100p149 = 0xc / 16.0 * p149 // = 0.1100p-149

	m0000p1074 = 0x0 / 16.0 * p1074 // = 0.0000p-1074
	m1000p1074 = 0x8 / 16.0 * p1074 // = 0.1000p-1074
	m1001p1074 = 0x9 / 16.0 * p1074 // = 0.1001p-1074
	m1011p1074 = 0xb / 16.0 * p1074 // = 0.1011p-1074
	m1100p1074 = 0xc / 16.0 * p1074 // = 0.1100p-1074
)

func main() {
	test32(float32(m0000p149), f32(m0000p149))
	test32(float32(m1000p149), f32(m1000p149))
	test32(float32(m1001p149), f32(m1001p149))
	test32(float32(m1011p149), f32(m1011p149))
	test32(float32(m1100p149), f32(m1100p149))

	test64(float64(m0000p1074), f64(m0000p1074))
	test64(float64(m1000p1074), f64(m1000p1074))
	test64(float64(m1001p1074), f64(m1001p1074))
	test64(float64(m1011p1074), f64(m1011p1074))
	test64(float64(m1100p1074), f64(m1100p1074))
}

func f32(x float64) float32 { return float32(x) }
func f64(x float64) float64 { return float64(x) }

func test32(a, b float32) {
	abits := math.Float32bits(a)
	bbits := math.Float32bits(b)
	if abits != bbits {
		panic(fmt.Sprintf("%08x != %08x\n", abits, bbits))
	}
}

func test64(a, b float64) {
	abits := math.Float64bits(a)
	bbits := math.Float64bits(b)
	if abits != bbits {
		panic(fmt.Sprintf("%016x != %016x\n", abits, bbits))
	}
}
