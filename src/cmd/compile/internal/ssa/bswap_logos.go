// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "math/bits"

// logosBswap requires a generic byte-swap opcode. It returns the permutation
// and its width mask, independent of the result type. In particular, Power10
// can represent Bswap16 with a wider uint result. Cast before swapping so
// sign-extended input facts cannot enter that result's upper bits.
func logosBswap(op Op, x int64) (value, mask int64) {
	switch op {
	case OpBswap16:
		return int64(bits.ReverseBytes16(uint16(x))), 0xffff
	case OpBswap32:
		return int64(bits.ReverseBytes32(uint32(x))), 0xffffffff
	case OpBswap64:
		return int64(bits.ReverseBytes64(uint64(x))), -1
	default:
		panic("unexpected byte-swap opcode: " + op.String())
	}
}

// A byte permutation moves each known bit to exactly one output position.
// Bits above the opcode width are zero, even when the result type is wider.
// fold applies the result type's normal sign extension after this transfer.
func (kb *knownBitsState) foldLogosBswap(v *Value) (value, known int64) {
	x, xk := kb.fold(v.Args[0])
	value, _ = logosBswap(v.Op, x)
	known, mask := logosBswap(v.Op, xk)
	return value, known | ^mask
}
