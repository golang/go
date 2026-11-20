// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"internal/cpu"
	"simd/archsimd"
)

// archClmulImpl returns how archsimd implements carryless multiply on vectors
// of the given bit size (see clmulImpl), and whether that is a single
// instruction on the whole vector. This mirrors the dispatch in archsimd's
// clmul_amd64.go: the full-width instruction if the CPU has it, otherwise
// PCLMULQDQ on each 128-bit part, otherwise software.
func archClmulImpl(size int) (impl string, full bool) {
	switch {
	case size == 256 && cpu.X86.HasVPCLMULQDQ:
		return "Uint64x4", true
	case size == 512 && cpu.X86.HasAVX512VPCLMULQDQ:
		return "Uint64x8", true
	case cpu.X86.HasPCLMULQDQ:
		return "Uint64x2", size == 128
	}
	return "software", false
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
