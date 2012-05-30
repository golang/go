// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark, taken from the shootuot, tests floating point performance.

package go1

import "testing"

func mandelbrot(n int) int {
	const Iter = 50
	const Zero float64 = 0
	const Limit = 2.0
	ok := 0
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			Zr, Zi, Tr, Ti := Zero, Zero, Zero, Zero
			Cr := (2*float64(x)/float64(n) - 1.5)
			Ci := (2*float64(y)/float64(n) - 1.0)

			for i := 0; i < Iter && (Tr+Ti <= Limit*Limit); i++ {
				Zi = 2*Zr*Zi + Ci
				Zr = Tr - Ti + Cr
				Tr = Zr * Zr
				Ti = Zi * Zi
			}

			if Tr+Ti <= Limit*Limit {
				ok++
			}
		}
	}
	return ok
}

func BenchmarkMandelbrot200(b *testing.B) {
	for i := 0; i < b.N; i++ {
		mandelbrot(200)
	}
}
