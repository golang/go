// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package simd

// These functions are declared once for all GOARCHes so that each has a single
// doc comment. The state they report is defined per GOARCH: in midway_common.go
// on GOARCHes with SIMD hardware support, and in simd_emulated.go elsewhere.

// VectorBits returns the bit length of the vectors used by this package.
//
// If [Emulated] is true, this is 128. Otherwise, it is the longest vector
// length supported by the hardware and the simd package. It can be artificially
// reduced by setting the GODEBUG=simd=<smaller size> environment variable
// before running a program.
func VectorBits() int {
	return maxVectorSize
}

// Emulated reports whether this package's operations are emulated in software
// rather than executed on vector hardware.
func Emulated() bool {
	return emulated
}
