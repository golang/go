// errorcheck -0 -m

//go:build goexperiment.simd && amd64

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "simd/archsimd"

func hasClosure(a, b, c, d archsimd.Int64x4) (w, x, y, z archsimd.Int64x4) {
	shuf := func() { // ERROR "can inline hasClosure.func1"
		w = z.RotateAllLeft(1).Xor(a)
		x = w.RotateAllLeft(3).Xor(b)
		y = x.RotateAllLeft(5).Xor(c)
		z = y.RotateAllLeft(7).Xor(d)
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), a.RotateAllLeft(1).Xor(d.RotateAllLeft(23))
		w = z.RotateAllLeft(1).Xor(a)
		x = w.RotateAllLeft(3).Xor(b)
		y = x.RotateAllLeft(5).Xor(c)
		z = y.RotateAllLeft(7).Xor(d)
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), a.RotateAllLeft(1).Xor(d.RotateAllLeft(23))
		w = z.RotateAllLeft(1).Xor(a)
		x = w.RotateAllLeft(3).Xor(b)
		y = x.RotateAllLeft(5).Xor(c)
		z = y.RotateAllLeft(7).Xor(d)
		a, b, c, d = b.RotateAllLeft(1).Xor(a.RotateAllLeft(23)), c.RotateAllLeft(1).Xor(b.RotateAllLeft(23)), d.RotateAllLeft(1).Xor(c.RotateAllLeft(23)), a.RotateAllLeft(1).Xor(d.RotateAllLeft(23))
	}

	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	shuf() // ERROR "inlining call to hasClosure.func1"
	return
}
