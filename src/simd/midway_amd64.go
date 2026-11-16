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

func archMaxVectorSize() (size, allFeatureSize int, arch string) {
	arch = "amd64"
	if archsimd.X86.AVX() {
		size = 128
	}
	if archsimd.X86.AVX2() {
		size = 256
	}
	if archsimd.X86.AVX512() {
		size = 512
	}
	// Carryless multiply doesn't limit the vector size: archsimd emulates
	// it on CPUs that lack it at their widest vector size.
	allFeatureSize = size
	// MulAdd requires FMA even for 128-bit vectors. Fall back to
	// emulation by default if it is unavailable.
	if !archsimd.X86.FMA() {
		allFeatureSize = 0
	}
	return
}
