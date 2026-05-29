// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd

// VectorBitSize returns the bit length of the longest vector available
// on the current hardware.  For arm64-neon, this is 128.
func VectorBitSize() int {
	return 128
}

// Emulated returns whether simd operations are emulated or
// running on actual vector hardware.
func Emulated() bool {
	return false
}
