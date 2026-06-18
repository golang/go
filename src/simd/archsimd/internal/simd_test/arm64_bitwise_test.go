// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestOrNot(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.OrNot, orNotSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.OrNot, orNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.OrNot, orNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.OrNot, orNotSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.OrNot, orNotSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.OrNot, orNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.OrNot, orNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.OrNot, orNotSlice[uint64])
}
