// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd

import (
	"internal/cpu"
)

const archHasHwClmul = true

func archMaxVectorSize() (size, allFeatureSize int) {
	// This describes Neon, SVE is still TBD.
	size = 128
	if cpu.ARM64.HasPMULL {
		allFeatureSize = 128
	}
	return
}
