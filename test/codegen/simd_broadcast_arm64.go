// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package codegen

import "simd/archsimd"

func broadcastInt8x16(x int8) archsimd.Int8x16 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.B16` -`VMOV`
	return archsimd.BroadcastInt8x16(x)
}

func broadcastUint8x16(x uint8) archsimd.Uint8x16 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.B16` -`VMOV`
	return archsimd.BroadcastUint8x16(x)
}

func broadcastInt16x8(x int16) archsimd.Int16x8 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.H8` -`VMOV`
	return archsimd.BroadcastInt16x8(x)
}

func broadcastUint16x8(x uint16) archsimd.Uint16x8 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.H8` -`VMOV`
	return archsimd.BroadcastUint16x8(x)
}

func broadcastInt32x4(x int32) archsimd.Int32x4 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.S4` -`VMOV`
	return archsimd.BroadcastInt32x4(x)
}

func broadcastUint32x4(x uint32) archsimd.Uint32x4 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.S4` -`VMOV`
	return archsimd.BroadcastUint32x4(x)
}

func broadcastInt64x2(x int64) archsimd.Int64x2 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.D2` -`VMOV`
	return archsimd.BroadcastInt64x2(x)
}

func broadcastUint64x2(x uint64) archsimd.Uint64x2 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.D2` -`VMOV`
	return archsimd.BroadcastUint64x2(x)
}

func broadcastFloat32x4(x float32) archsimd.Float32x4 {
	// arm64:`VDUP V[0-9]+\.S\[0\], V[0-9]+\.S4` -`VMOV`
	return archsimd.BroadcastFloat32x4(x)
}

func broadcastFloat64x2(x float64) archsimd.Float64x2 {
	// arm64:`VDUP V[0-9]+\.D\[0\], V[0-9]+\.D2` -`VMOV`
	return archsimd.BroadcastFloat64x2(x)
}

func broadcastLoadFloat32x4(p *float32) archsimd.Float32x4 {
	// arm64:`FMOVS` `VDUP V[0-9]+\.S\[0\], V[0-9]+\.S4` -`VMOV`
	return archsimd.BroadcastFloat32x4(*p)
}

func broadcastLoadFloat64x2(p *float64) archsimd.Float64x2 {
	// arm64:`FMOVD` `VDUP V[0-9]+\.D\[0\], V[0-9]+\.D2` -`VMOV`
	return archsimd.BroadcastFloat64x2(*p)
}

// ShiftAll broadcasts a scalar after inserting it into the input vector.
func broadcastShiftInt8x16(x archsimd.Int8x16, n uint64) archsimd.Int8x16 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.B16` `VSSHL` -`VMOV`
	return x.ShiftAllLeft(n)
}

func broadcastShiftUint16x8(x archsimd.Uint16x8, n uint64) archsimd.Uint16x8 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.H8` `VUSHL` -`VMOV`
	return x.ShiftAllRight(n)
}

func broadcastShiftInt32x4(x archsimd.Int32x4, n uint64) archsimd.Int32x4 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.S4` `VSSHL` -`VMOV`
	return x.ShiftAllRight(n)
}

func broadcastShiftUint64x2(x archsimd.Uint64x2, n uint64) archsimd.Uint64x2 {
	// arm64:`VDUP R[0-9]+, V[0-9]+\.D2` `VUSHL` -`VMOV`
	return x.ShiftAllLeft(n)
}
