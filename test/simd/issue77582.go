// compile

//go:build goexperiment.simd && amd64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"simd/archsimd"
)

func vpternlogdPanic() {
	resultsMask := archsimd.Mask64x8{}
	a := archsimd.Mask64x8FromBits(0xFF)
	b := archsimd.Float64x8{}.Less(archsimd.Float64x8{})

	for i := 0; i < 1; i++ {
		resultsMask = a.Or(b).Or(resultsMask)
		// This nested logic triggered the panic in CL 745460
		_ = resultsMask.And(resultsMask.And(archsimd.Mask64x8{}))
	}
}
