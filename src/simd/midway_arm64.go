// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd

import (
	"internal/cpu"
)

// archHasHwClmul reports whether carryless multiply on vectors of the given
// bit size is a single hardware instruction. Without PMULL, archsimd emulates
// it using other Neon instructions.
func archHasHwClmul(size int) bool {
	return cpu.ARM64.HasPMULL
}

func archMaxVectorSize() (size, allFeatureSize int, arch string) {
	arch = "arm64"
	// This describes Neon, SVE is still TBD.
	//
	// All operations are available with just Neon. In particular, archsimd
	// emulates carryless multiply on CPUs without PMULL.
	return 128, 128, arch
}
