// compile

//go:build amd64 && goexperiment.simd

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for rematerialization ignoring the register constraint
// during regalloc's shuffle phase.

package p

import (
	"simd"
)

func PackComplex(b bool) {
	for {
		if b {
			var indices [4]uint32
			simd.Uint32x4{}.ShiftAllRight(20).Store(&indices)
			_ = indices[indices[0]]
		}
	}
}
