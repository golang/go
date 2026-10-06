// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package p

import "simd/archsimd"

//go:noinline
func fu() {}

func f(z archsimd.Float32x8, m archsimd.Mask32x8, n int) archsimd.Float32x8 {
	_ = func() {
		z = z.Compress(m)
		fu() // m is live across the call, so m is spilled
		for range n {
			z = z.Compress(m) // m is reloaded
		}
	}
	return z
}
