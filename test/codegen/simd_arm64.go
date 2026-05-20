// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check ARM64 SIMD code generation and peephole optimizations.

//go:build goexperiment.simd && arm64

package codegen

import (
	"simd/archsimd"
)

//go:noinline
func forceSpill() {}

func spillAroundCall(a archsimd.Int8x16) archsimd.Int8x16 {
	forceSpill()
	// arm64:`FMOVQ` `FMOVQ`
	return a
}
