// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// This exposes some internal interfaces to archsimd_test.

package archsimd

func (x Uint8x16) ExportTestCarrylessMultiplyWidenLo(y Uint8x16) Uint16x8 {
	return x.carrylessMultiplyWidenLo(y)
}

func (x Uint8x16) ExportTestCarrylessMultiplyWidenHi(y Uint8x16) Uint16x8 {
	return x.HiToLo().carrylessMultiplyWidenLo(y.HiToLo())
}
