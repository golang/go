// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package bridge

import "simd/archsimd"

func (x Float32x4) ToArch() any {
	return archsimd.Float32x4(x)
}

func (x Float64x2) ToArch() any {
	return archsimd.Float64x2(x)
}

func (x Int16x8) ToArch() any {
	return archsimd.Int16x8(x)
}

func (x Int32x4) ToArch() any {
	return archsimd.Int32x4(x)
}

func (x Int64x2) ToArch() any {
	return archsimd.Int64x2(x)
}

func (x Int8x16) ToArch() any {
	return archsimd.Int8x16(x)
}

func (x Mask16x8) ToArch() any {
	return archsimd.Mask16x8(x)
}

func (x Mask32x4) ToArch() any {
	return archsimd.Mask32x4(x)
}

func (x Mask64x2) ToArch() any {
	return archsimd.Mask64x2(x)
}

func (x Mask8x16) ToArch() any {
	return archsimd.Mask8x16(x)
}

func (x Uint16x8) ToArch() any {
	return archsimd.Uint16x8(x)
}

func (x Uint32x4) ToArch() any {
	return archsimd.Uint32x4(x)
}

func (x Uint64x2) ToArch() any {
	return archsimd.Uint64x2(x)
}

func (x Uint8x16) ToArch() any {
	return archsimd.Uint8x16(x)
}
