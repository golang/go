// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64 && linux

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// See slicepart_boundary_128_test.go for testLoadPartPageBoundary.

func TestLoadPartPageBoundary256(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
	}
	testLoadPartPageBoundary(t, archsimd.LoadInt8x32Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt16x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt32x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt64x4Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint8x32Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint16x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint32x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint64x4Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat32x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat64x4Part)
}

func TestLoadPartPageBoundary512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	testLoadPartPageBoundary(t, archsimd.LoadInt8x64Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt16x32Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt32x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt64x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint8x64Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint16x32Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint32x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint64x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat32x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat64x8Part)
}
