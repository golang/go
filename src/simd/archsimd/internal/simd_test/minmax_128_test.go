// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package simd_test

import (
	"runtime"
	"simd/archsimd"
)

func floatMinMaxPlatformConfig() floatMinMaxConfig {
	return floatMinMaxConfig{allowMasked: true, allowAnyNaN: runtime.GOARCH == "wasm"}
}

func floatMinMaxMask32x4() archsimd.Mask32x4 {
	return archsimd.LoadInt32x4([]int32{1, 0, 1, 0}).Greater(archsimd.LoadInt32x4([]int32{0, 0, 0, 0}))
}

func floatMinMaxMask64x2() archsimd.Mask64x2 {
	return archsimd.LoadInt64x2([]int64{1, 0}).Greater(archsimd.LoadInt64x2([]int64{0, 0}))
}
