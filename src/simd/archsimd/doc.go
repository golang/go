// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// Package archsimd provides access to architecture-specific SIMD operations.
//
// This is a low-level package that exposes hardware-specific functionality.
// It currently supports AMD64.
//
// This package is experimental, and not subject to the Go 1 compatibility promise.
// It only exists when building with the GOEXPERIMENT=simd environment variable set.
//
// # Vector types and operations
//
// Vector types are defined as structs, such as Int8x16 and Float64x8, corresponding
// to the hardware's vector registers. On AMD64, 128-, 256-, and 512-bit vectors are
// supported.
//
// Mask types are defined similarly, such as Mask8x16, and are represented as
// opaque types, handling the differences in the underlying representations.
// A mask can be converted to/from the corresponding integer vector type, or
// to/from a bitmask.
//
// Operations are mostly defined as methods on the vector types. Most of them
// are compiler intrinsics and correspond directly to hardware instructions.
//
// Common operations include:
//   - Load/Store: Load a vector from memory or store a vector to memory.
//   - Arithmetic: Add, Sub, Mul, etc.
//   - Bitwise: And, Or, Xor, etc.
//   - Comparison: Equal, Greater, etc., which produce a mask.
//   - Conversion: Convert between different vector types.
//   - Field selection and rearrangement: GetElem, Permute, etc.
//   - Masking: Masked, Merge.
//
// The compiler recognizes certain patterns of operations and may optimize
// them to more performant instructions. For example, on AVX512, an Add operation
// followed by Masked may be optimized to a masked add instruction.
// For this reason, not all hardware instructions are available as APIs.
//
// # CPU feature checks
//
// The package provides global variables to check for CPU features available
// at runtime. For example, on AMD64, the [X86] variable provides methods to
// check for AVX2, AVX512, etc.
// It is recommended to check for CPU features before using the corresponding
// vector operations.
//
// # Notes
//
//   - This package is not portable, as the available types and operations depend
//     on the target architecture. It is not recommended to expose the SIMD types
//     defined in this package in public APIs.
//   - For performance reasons, it is recommended to use the vector types directly
//     as values. It is not recommended to take the address of a vector type,
//     allocate it in the heap, or put it in an aggregate type.
package archsimd

// BUG(cherry): Using a vector type as a type parameter may not work.

// BUG(cherry): Using reflect Call to call a vector function/method may not work.
