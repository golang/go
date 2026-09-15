// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"simd/internal/test_helpers"
	"testing"
)

func TestMask8x16Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt8x16)
}

func TestMask8x32Range(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt8x32)
}

func TestMask8x64Range(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt8x64)
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

func TestMask16x16Range(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt16x16)
}

func TestMask32x8Range(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt32x8)
}

func TestMask64x4Range(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt64x4)
}

func TestMask16x32Range(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt16x32)
}

func TestMask32x16Range(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt32x16)
}

func TestMask64x8Range(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 not supported")
	}
	test_helpers.TestMaskAllAny(t, archsimd.LoadInt64x8)
}
