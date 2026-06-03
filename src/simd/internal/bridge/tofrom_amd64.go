// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package bridge

import "simd/archsimd"

// For amd64, handle the larger types not mentioned in tofrom_128.go

func (x Float32x16) ToArch() any {
	return archsimd.Float32x16(x)
}

func (x Float32x8) ToArch() any {
	return archsimd.Float32x8(x)
}

func (x Float64x4) ToArch() any {
	return archsimd.Float64x4(x)
}

func (x Float64x8) ToArch() any {
	return archsimd.Float64x8(x)
}

func (x Int16x16) ToArch() any {
	return archsimd.Int16x16(x)
}

func (x Int16x32) ToArch() any {
	return archsimd.Int16x32(x)
}

func (x Int32x16) ToArch() any {
	return archsimd.Int32x16(x)
}

func (x Int32x8) ToArch() any {
	return archsimd.Int32x8(x)
}

func (x Int64x4) ToArch() any {
	return archsimd.Int64x4(x)
}

func (x Int64x8) ToArch() any {
	return archsimd.Int64x8(x)
}

func (x Int8x32) ToArch() any {
	return archsimd.Int8x32(x)
}

func (x Int8x64) ToArch() any {
	return archsimd.Int8x64(x)
}

func (x Mask16x16) ToArch() any {
	return archsimd.Mask16x16(x)
}

func (x Mask16x32) ToArch() any {
	return archsimd.Mask16x32(x)
}

func (x Mask32x16) ToArch() any {
	return archsimd.Mask32x16(x)
}

func (x Mask32x8) ToArch() any {
	return archsimd.Mask32x8(x)
}

func (x Mask64x4) ToArch() any {
	return archsimd.Mask64x4(x)
}

func (x Mask64x8) ToArch() any {
	return archsimd.Mask64x8(x)
}

func (x Mask8x32) ToArch() any {
	return archsimd.Mask8x32(x)
}

func (x Mask8x64) ToArch() any {
	return archsimd.Mask8x64(x)
}

func (x Uint16x16) ToArch() any {
	return archsimd.Uint16x16(x)
}

func (x Uint16x32) ToArch() any {
	return archsimd.Uint16x32(x)
}

func (x Uint32x16) ToArch() any {
	return archsimd.Uint32x16(x)
}

func (x Uint32x8) ToArch() any {
	return archsimd.Uint32x8(x)
}

func (x Uint64x4) ToArch() any {
	return archsimd.Uint64x4(x)
}

func (x Uint64x8) ToArch() any {
	return archsimd.Uint64x8(x)
}

func (x Uint8x32) ToArch() any {
	return archsimd.Uint8x32(x)
}

func (x Uint8x64) ToArch() any {
	return archsimd.Uint8x64(x)
}
