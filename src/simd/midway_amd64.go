// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"internal/cpu"
	"simd/archsimd"
)

const archHasHwClmul = true

func archMaxVectorSize() (size, allFeatureSize int) {
	if archsimd.X86.AVX() {
		size = 128
		allFeatureSize = 128
	}
	if archsimd.X86.AVX2() {
		size = 256
		if cpu.X86.HasVPCLMULQDQ {
			allFeatureSize = 256
		}
	}
	if archsimd.X86.AVX512() {
		size = 512
		if cpu.X86.HasAVX512VPCLMULQDQ {
			allFeatureSize = 512
		}
	}
	return
}
