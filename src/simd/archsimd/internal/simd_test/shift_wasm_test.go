// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestShiftAll8(t *testing.T) {
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllLeft, shiftAllLeftSlice[int8])
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllLeft, shiftAllLeftSlice[uint8])
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllRight, shiftAllRightSlice[int8])
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllRight, shiftAllRightSlice[uint8])
}
