// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"internal/cpu"
	"simd/archsimd"
)

// archHasHwClmul reports whether carryless multiply on vectors of the given
// bit size is a single hardware instruction.
func archHasHwClmul(size int) bool {
	switch size {
	case 128:
		return cpu.X86.HasPCLMULQDQ
	case 256:
		return cpu.X86.HasVPCLMULQDQ
	case 512:
		return cpu.X86.HasAVX512VPCLMULQDQ
	}
	return false
}

// archMaxVectorSize returns the bit size of the largest vectors the CPU
// supports, or 0 if it doesn't support the SIMD features this package
// requires.
func archMaxVectorSize() (size int) {
	if archsimd.X86.AVX() {
		size = 128
	}
	if archsimd.X86.AVX2() {
		size = 256
	}
	if archsimd.X86.AVX512() {
		size = 512
	}
	// MulAdd requires FMA even for 128-bit vectors, so without FMA the
	// package can't use SIMD hardware at all.
	if !archsimd.X86.FMA() {
		size = 0
	}
	return size
}
