// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && arm64

package simd

import (
	"internal/cpu"
)

// archClmulImpl returns how archsimd implements carryless multiply on vectors
// of the given bit size (see clmulImpl), and whether that is a single
// instruction on the whole vector. This mirrors the dispatch in archsimd's
// clmul_arm64.go: the 64-bit PMULL if the CPU has it, otherwise an emulation
// using the 8-bit PMULL on Uint8x16. Vectors are always 128 bits.
func archClmulImpl(size int) (impl string, full bool) {
	if cpu.ARM64.HasPMULL {
		return "Uint64x2", true
	}
	return "Uint8x16", false
}

// archMaxVectorSize returns the bit size of the largest vectors the CPU
// supports.
//
// This describes Neon, SVE is still TBD.
func archMaxVectorSize() (size int) {
	return 128
}
