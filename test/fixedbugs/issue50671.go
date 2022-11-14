// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 50671: sign extension eliminated incorrectly on MIPS64.

package main

//go:noinline
func F(x int32) (float64, int64) {
	a := float64(x)
	b := int64(x)
	return a, b
}

var a, b, c float64

// Poison some floating point registers with non-zero high bits.
//
//go:noinline
func poison(x float64) {
	a = x - 123.45
	b = a * 1.2
	c = b + 3.4
}

func main() {
	poison(333.3)
	_, b := F(123)
	if b != 123 {
		panic("FAIL")
	}
}
