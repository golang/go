// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12577: Test that there are no -0 floating-point constants.

package main

import "math"

const (
	z0 = 0.0
	z1 = -0.0
	z2 = -z0
	z3 = -z2
)

var (
	x0 float32 = z0
	x1 float32 = z1
	x2 float32 = z2
	x3 float32 = z3

	y0 float64 = z0
	y1 float64 = z1
	y2 float64 = z2
	y3 float64 = z3
)

func test32(f float32) {
	if f != 0 || math.Signbit(float64(f)) {
		println("BUG: got", f, "want 0.0")
		return
	}
}

func test64(f float64) {
	if f != 0 || math.Signbit(f) {
		println("BUG: got", f, "want 0.0")
		return
	}
}

func main() {
	if f := -x0; f != 0 || !math.Signbit(float64(f)) {
		println("BUG: got", f, "want -0.0")
	}

	test32(-0.0)
	test32(x0)
	test32(x1)
	test32(x2)
	test32(x3)

	if f := -y0; f != 0 || !math.Signbit(f) {
		println("BUG: got", f, "want -0.0")
	}

	test64(-0.0)
	test64(y0)
	test64(y1)
	test64(y2)
	test64(y3)
}
