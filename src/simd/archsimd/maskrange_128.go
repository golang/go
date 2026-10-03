// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package archsimd

import "math/bits"

// All returns true when all positions in mask x are true.
//
// Emulated: GetElem
func (x Mask8x16) All() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0101010101010101 == 0x0101010101010101
}

// Any returns true when any position in mask x is true.
//
// Emulated: GetElem
func (x Mask8x16) Any() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0101010101010101 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated: GetElem
func (x Mask8x16) None() bool {
	word := x.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0101010101010101 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated: GetElem
func (x Mask16x8) All() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0001000100010001 == 0x0001000100010001
}

// Any returns true when any position in mask x is true.
//
// Emulated: GetElem
func (x Mask16x8) Any() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0001000100010001 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated: GetElem
func (x Mask16x8) None() bool {
	word := x.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0001000100010001 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated: GetElem
func (x Mask32x4) All() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0&b0)&0x0000000100000001 == 0x0000000100000001
}

// Any returns true when any position in mask x is true.
//
// Emulated: GetElem
func (x Mask32x4) Any() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0000000100000001 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated: GetElem
func (x Mask32x4) None() bool {
	word := x.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return (a0|b0)&0x0000000100000001 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated: GetElem
func (x Mask64x2) All() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 != 0 && b0 != 0
}

// Any returns true when any position in mask x is true.
//
// Emulated: GetElem
func (x Mask64x2) Any() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 != 0 || b0 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated: GetElem
func (x Mask64x2) None() bool {
	word := x.ToInt64x2().ToBits()
	a0 := word.GetElem(0)
	b0 := word.GetElem(1)
	return a0 == 0 && b0 == 0
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated: GetElem
func (m Mask8x16) TrailingZeros() int {
	word := m.ToInt8x16().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	a := a0 & 0x0101010101010101
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 3
	}
	a0 = word.GetElem(1)
	a = a0 & 0x0101010101010101
	lane = bits.TrailingZeros64(a)
	return lane>>3 + 8
}

// TrailingZeros returns the number of trailing (low-order) zeroes in mask m
//
// Emulated: GetElem
func (m Mask16x8) TrailingZeros() int {
	word := m.ToInt16x8().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	a := a0 & 0x0001000100010001
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 4
	}
	a0 = word.GetElem(1)
	a = a0 & 0x0001000100010001
	lane = bits.TrailingZeros64(a)
	return lane>>4 + 4
}

// TrailingZeros returns the number of trailing (low-order) zeroes in mask m
//
// Emulated: GetElem
func (m Mask32x4) TrailingZeros() int {
	word := m.ToInt32x4().ToBits().ReshapeToUint64s()
	a0 := word.GetElem(0)
	a := a0 & 0x0000000100000001
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 5
	}
	a0 = word.GetElem(1)
	a = a0 & 0x0000000100000001
	lane = bits.TrailingZeros64(a)
	return lane>>5 + 2
}

// TrailingZeros returns the number of trailing (low-order) zeroes in mask m
//
// Emulated: GetElem
func (m Mask64x2) TrailingZeros() int {
	word := m.ToInt64x2().ToBits()
	if word.GetElem(0) != 0 {
		return 0
	}
	if word.GetElem(1) != 0 {
		return 1
	}
	return 2
}
