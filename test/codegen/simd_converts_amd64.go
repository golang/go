// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of amd64 SIMD conversions.

//go:build goexperiment.simd && amd64

package codegen

import "simd/archsimd"

// A widening conversion reads only the low part of its input, so a load of
// the input folds into it, whatever the input's width.

func simdExtendToInt32LoadFold(s []int8) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD \([A-Z0-9]+\), Y[0-9]+`
	return archsimd.LoadInt8x32(s).ExtendToInt32()
}

func simdExtendLo8ToInt32LoadFold(s []int8) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD \([A-Z0-9]+\), Y[0-9]+`
	return archsimd.LoadInt8x16(s).ExtendLo8ToInt32()
}
