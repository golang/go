// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package archsimd

// All returns true when all positions in mask x are true.
//
// Emulated
func (x Mask8x16) All() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0101010101010101 == 0x0101010101010101
}

// Any returns true when any position in mask x is true.
//
// Emulated
func (x Mask8x16) Any() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0101010101010101 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated
func (x Mask8x16) None() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0101010101010101 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated
func (x Mask16x8) All() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0001000100010001 == 0x0001000100010001
}

// Any returns true when any position in mask x is true.
//
// Emulated
func (x Mask16x8) Any() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0001000100010001 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated
func (x Mask16x8) None() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0001000100010001 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated
func (x Mask32x4) All() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0000000100000001 == 0x0000000100000001
}

// Any returns true when any position in mask x is true.
//
// Emulated
func (x Mask32x4) Any() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0000000100000001 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated
func (x Mask32x4) None() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0000000100000001 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated
func (x Mask64x2) All() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 != 0 && b0 != 0
}

// Any returns true when any position in mask x is true.
//
// Emulated
func (x Mask64x2) Any() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 != 0 || b0 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated
func (x Mask64x2) None() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 == 0 && b0 == 0
}
