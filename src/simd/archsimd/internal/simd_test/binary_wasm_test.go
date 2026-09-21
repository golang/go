// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestMaxWasm(t *testing.T) {
	testInt64x2Binary(t, archsimd.Int64x2.Max, maxSlice[int64])
	testUint64x2Binary(t, archsimd.Uint64x2.Max, maxSlice[uint64])
}

func TestMinWasm(t *testing.T) {
	testInt64x2Binary(t, archsimd.Int64x2.Min, minSlice[int64])
	testUint64x2Binary(t, archsimd.Uint64x2.Min, minSlice[uint64])
}

func TestMulWasm(t *testing.T) {
	testInt64x2Binary(t, archsimd.Int64x2.Mul, mulSlice[int64])
	testUint64x2Binary(t, archsimd.Uint64x2.Mul, mulSlice[uint64])
}
