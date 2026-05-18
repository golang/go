// errorcheck -0 -m

//go:build goexperiment.simd && amd64

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "simd/archsimd"

func hasClosure(a, b, c, d archsimd.Int64x4) (w, x, y, z archsimd.Int64x4) {
	shuf := func() { // ERROR "can inline hasClosure.func1"
		w = z.RotateAllLeft(1).Xor(a)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		x = w.RotateAllLeft(3).Xor(b)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		y = x.RotateAllLeft(5).Xor(c)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		z = y.RotateAllLeft(7).Xor(d)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			a.RotateAllLeft(1).Xor(d.RotateAllLeft(23)) // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		w = z.RotateAllLeft(1).Xor(a)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		x = w.RotateAllLeft(3).Xor(b)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		y = x.RotateAllLeft(5).Xor(c)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		z = y.RotateAllLeft(7).Xor(d)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			a.RotateAllLeft(1).Xor(d.RotateAllLeft(23)) // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		w = z.RotateAllLeft(1).Xor(a)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		x = w.RotateAllLeft(3).Xor(b)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		y = x.RotateAllLeft(5).Xor(c)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		z = y.RotateAllLeft(7).Xor(d)                             // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
			a.RotateAllLeft(1).Xor(d.RotateAllLeft(23)) // ERROR "inlining call to archsimd.Int64x4.RotateAllLeft"
	}

	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	shuf() // ERROR "inlining call to hasClosure.func1" "inlining call to archsimd.Int64x4.RotateAllLeft"
	return
}
