// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd

// Deprecated APIs that are wrappers around their replacements.

// VectorBitSize returns the bit length of the vectors used by this package.
//
// Deprecated: use [VectorBits] instead.
//
//go:fix inline
func VectorBitSize() int {
	return VectorBits()
}
