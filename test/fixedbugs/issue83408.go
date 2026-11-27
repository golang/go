// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func g() {}

//go:noinline
func f(c bool, x float64, y float32) (int16, uint8) {
	a, b := int16(7), uint8(7)
	if c {
		a, b = int16(x), uint8(y)
	}
	g()
	return a, b
}

func main() {
	a, b := f(true, 1000, 200)
	if a != 1000 {
		panic(a)
	}
	if b != 200 {
		panic(b)
	}
}
