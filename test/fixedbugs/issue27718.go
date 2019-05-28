// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// (-0)+0 should be 0, not -0.

package main

//go:noinline
func add64(x float64) float64 {
	return x + 0
}

func testAdd64() {
	var zero float64
	inf := 1.0 / zero
	negZero := -1 / inf
	if 1/add64(negZero) != inf {
		panic("negZero+0 != posZero (64 bit)")
	}
}

//go:noinline
func sub64(x float64) float64 {
	return x - 0
}

func testSub64() {
	var zero float64
	inf := 1.0 / zero
	negZero := -1 / inf
	if 1/sub64(negZero) != -inf {
		panic("negZero-0 != negZero (64 bit)")
	}
}

//go:noinline
func add32(x float32) float32 {
	return x + 0
}

func testAdd32() {
	var zero float32
	inf := 1.0 / zero
	negZero := -1 / inf
	if 1/add32(negZero) != inf {
		panic("negZero+0 != posZero (32 bit)")
	}
}

//go:noinline
func sub32(x float32) float32 {
	return x - 0
}

func testSub32() {
	var zero float32
	inf := 1.0 / zero
	negZero := -1 / inf
	if 1/sub32(negZero) != -inf {
		panic("negZero-0 != negZero (32 bit)")
	}
}

func main() {
	testAdd64()
	testSub64()
	testAdd32()
	testSub32()
}
