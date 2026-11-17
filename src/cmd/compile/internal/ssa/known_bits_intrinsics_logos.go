// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "math/bits"

// foldLogosBitIntrinsic handles the unary BitLen, Ctz, and PopCount operations.
// Every feasible input contains all bits in ones and no bits outside possible.
// Masking to the opcode width removes fold's canonical sign extension and any
// wider input bits. The result width is independent and normalized by fold.
func (kb *knownBitsState) foldLogosBitIntrinsic(v *Value) (value, known int64) {
	var width int
	switch v.Op {
	case OpBitLen8, OpCtz8, OpCtz8NonZero, OpPopCount8:
		width = 8
	case OpBitLen16, OpCtz16, OpCtz16NonZero, OpPopCount16:
		width = 16
	case OpBitLen32, OpCtz32, OpCtz32NonZero, OpPopCount32:
		width = 32
	case OpBitLen64, OpCtz64, OpCtz64NonZero, OpPopCount64:
		width = 64
	default:
		panic("unreachable; unsupported bit intrinsic")
	}
	input := v.Args[0]
	mask := ^uint64(0) >> (64 - width)
	x, xk := kb.fold(input)
	ones := uint64(x) & mask
	possible := (uint64(x) | ^uint64(xk)) & mask
	var lo, hi int
	switch v.Op {
	case OpBitLen8, OpBitLen16, OpBitLen32, OpBitLen64:
		lo, hi = bits.Len64(ones), bits.Len64(possible)
	case OpCtz8, OpCtz16, OpCtz32, OpCtz64,
		OpCtz8NonZero, OpCtz16NonZero, OpCtz32NonZero, OpCtz64NonZero:
		// More possible one bits can only reduce the trailing-zero count.
		// Keep zero-inclusive bounds even for NonZero: its path-dependent
		// precondition need not be visible in this bit lattice.
		lo = min(bits.TrailingZeros64(possible), width)
		hi = min(bits.TrailingZeros64(ones), width)
	case OpPopCount8, OpPopCount16, OpPopCount32, OpPopCount64:
		lo, hi = bits.OnesCount64(ones), bits.OnesCount64(possible)
	default:
		panic("unreachable; unsupported bit intrinsic")
	}
	// All results lie in [lo, hi]. Bits above the highest differing bit
	// of the endpoints form a common prefix throughout that interval.
	// Both endpoints are at most 64, so the shift is at most 7.
	known = -1 << bits.Len64(uint64(lo^hi))
	return int64(lo) & known, known
}
