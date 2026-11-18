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

// HasHardwareCarrylessMultiply reports whether the CarrylessMul methods
// of [Uint64s] are a single hardware instruction at the vector size this
// package uses.
//
// Deprecated: The CarrylessMul methods are safe to call on every CPU, so
// there is no need to check this before calling them. If the CPU lacks a
// carryless multiply instruction at the vector size in use, they are
// implemented with narrower instructions or software. Code that selects an
// algorithm based on whether carryless multiply is fast should instead check
// the CPU features in [simd/archsimd] that the CarrylessMul methods
// there document.
func HasHardwareCarrylessMultiply() bool {
	return hwClmul
}
