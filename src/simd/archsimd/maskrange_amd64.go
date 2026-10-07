// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

import "math/bits"

// 128-bit masks

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX
func (x Mask8x16) All() bool {
	return x.ToBits() == 0xffff
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX
func (x Mask8x16) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX
func (x Mask8x16) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX
func (x Mask16x8) All() bool {
	return x.ToInt16x8().AsInt8x16().asMask().ToBits()&0x5555 == 0x5555
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX
func (x Mask16x8) Any() bool {
	return x.ToInt16x8().AsInt8x16().asMask().ToBits()&0x5555 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX
func (x Mask16x8) None() bool {
	return x.ToInt16x8().AsInt8x16().asMask().ToBits()&0x5555 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX
func (x Mask32x4) All() bool {
	return x.ToBits()&0x0f == 0x0f
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX
func (x Mask32x4) Any() bool {
	return x.ToBits()&0x0f != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX
func (x Mask32x4) None() bool {
	return x.ToBits()&0x0f == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX
func (x Mask64x2) All() bool {
	return x.ToBits()&0x03 == 0x03
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX
func (x Mask64x2) Any() bool {
	return x.ToBits()&0x03 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX
func (x Mask64x2) None() bool {
	return x.ToBits()&0x03 == 0
}

// 256-bit masks

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX2
func (x Mask8x32) All() bool {
	return x.ToBits() == 0xffffffff
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX2
func (x Mask8x32) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX2
func (x Mask8x32) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX2
func (x Mask16x16) All() bool {
	return x.ToInt16x16().AsInt8x32().asMask().ToBits()&0x55555555 == 0x55555555
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX2
func (x Mask16x16) Any() bool {
	return x.ToInt16x16().AsInt8x32().asMask().ToBits()&0x55555555 != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX2
func (x Mask16x16) None() bool {
	return x.ToInt16x16().AsInt8x32().asMask().ToBits()&0x55555555 == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX2
func (x Mask32x8) All() bool {
	return x.ToBits() == 0xff
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX2
func (x Mask32x8) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX2
func (x Mask32x8) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX2
func (x Mask64x4) All() bool {
	return x.ToBits()&0x0f == 0x0f
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX2
func (x Mask64x4) Any() bool {
	return x.ToBits()&0x0f != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX2
func (x Mask64x4) None() bool {
	return x.ToBits()&0x0f == 0
}

// 512-bit masks

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX512
func (x Mask8x64) All() bool {
	return ^x.ToBits() == 0
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX512
func (x Mask8x64) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX512
func (x Mask8x64) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX512
func (x Mask16x32) All() bool {
	return ^x.ToBits() == 0
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX512
func (x Mask16x32) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX512
func (x Mask16x32) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX512
func (x Mask32x16) All() bool {
	return ^x.ToBits() == 0
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX512
func (x Mask32x16) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX512
func (x Mask32x16) None() bool {
	return x.ToBits() == 0
}

// All returns true when all positions in mask x are true.
//
// Emulated, CPU Feature AVX512
func (x Mask64x8) All() bool {
	return ^x.ToBits() == 0
}

// Any returns true when any position in mask x is true.
//
// Emulated, CPU Feature AVX512
func (x Mask64x8) Any() bool {
	return x.ToBits() != 0
}

// None returns true when no positions in mask x are set.
//
// Emulated, CPU Feature AVX512
func (x Mask64x8) None() bool {
	return x.ToBits() == 0
}

// TrailingZeros returns the number of low-order false (zero) elements in mask x.
//
// Emulated, CPU Feature AVX
func (x Mask8x16) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 16 {
		return 16
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask x.
//
// Emulated, CPU Feature AVX
func (x Mask8x32) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 32 {
		return 32
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask x.
//
// Emulated, CPU Feature AVX
func (x Mask8x64) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 64 {
		return 64
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX
func (x Mask16x8) TrailingZeros() int {
	word := uint64(x.ToInt16x8().AsInt8x16().asMask().ToBits() & 0x5555)
	lane := bits.TrailingZeros64(word) >> 1
	if lane >= 8 {
		return 8
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX2
func (x Mask16x16) TrailingZeros() int {
	word := uint64(x.ToInt16x16().AsInt8x32().asMask().ToBits() & 0x55555555)
	lane := bits.TrailingZeros64(word) >> 1
	if lane >= 16 {
		return 16
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX512
func (x Mask16x32) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 32 {
		return 32
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX
func (x Mask32x4) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 4 {
		return 4
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX2
func (x Mask32x8) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 8 {
		return 8
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX512
func (x Mask32x16) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 16 {
		return 16
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX
func (x Mask64x2) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 2 {
		return 2
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX2
func (x Mask64x4) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 4 {
		return 4
	}
	return lane
}

// TrailingZeros returns the number of low-order false (zero) elements in mask m.
//
// Emulated, CPU Feature AVX512
func (x Mask64x8) TrailingZeros() int {
	word := uint64(x.ToBits())
	lane := bits.TrailingZeros64(word)
	if lane >= 8 {
		return 8
	}
	return lane
}
