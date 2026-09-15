// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package simd_test

import (
	"simd/archsimd"
	"simd/internal/test_helpers"
	"testing"
)

func TestMask8x16Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt8x16)
}

func TestMask16x8Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt16x8)
}

func TestMask32x4Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt32x4)
}

func TestMask64x2Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt64x2)
}
