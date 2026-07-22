// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssacore"
	"math/bits"
)

func getPPC64ShiftMaskLength(v int64) int64 {
	return int64(bits.Len64(uint64(v)))
}

// Test if this mask is a valid, contiguous bitmask which can be
// represented by a RLWNM mask and also clears the upper 32 bits
// of the register.
func isPPC64WordRotateMaskNonWrapping(v64 int64) bool {
	// Isolate rightmost 1 (if none 0) and add.
	v := uint32(v64)
	vp := (v & -v) + v
	return (v&vp == 0) && v != 0 && uint64(uint32(v64)) == uint64(v64)
}

// isU16Bit reports whether n can be represented as an unsigned 16 bit integer.
func isU16Bit(n int64) bool {
	return n == int64(uint16(n))
}

// Test if RLWINM feeding into an ANDconst can be merged. Return the encoded RLWINM constant,
// or 0 if they cannot be merged.
func mergePPC64AndRlwinm(mask uint32, rlw int64) int64 {
	r, _, _, mask_rlw := ssacore.DecodePPC64RotateMask(rlw)
	mask_out := (mask_rlw & uint64(mask))

	// Verify the result is still a valid bitmask of <= 32 bits.
	if !IsPPC64WordRotateMask(int64(mask_out)) {
		return 0
	}
	return EncodePPC64RotateMask(r, int64(mask_out), 32)
}

// Combine (ANDconst [m] (SLDconst [s])) into (RLWINM [y]) or return 0
func mergePPC64AndSldi(m, s int64) int64 {
	mask := -1 << s & m

	// Verify the rotate and mask result only uses the lower 32 bits.
	rv := bits.RotateLeft64(0xFFFFFFFF00000000, int(s))
	if rv&uint64(mask) != 0 {
		return 0
	}
	if !isPPC64WordRotateMaskNonWrapping(mask) {
		return 0
	}
	return EncodePPC64RotateMask(s&31, mask, 32)
}

// Combine (ANDconst [m] (SRDconst [s])) into (RLWINM [y]) or return 0
func mergePPC64AndSrdi(m, s int64) int64 {
	mask := MergePPC64RShiftMask(m, s, 64)

	// Verify the rotate and mask result only uses the lower 32 bits.
	rv := bits.RotateLeft64(0xFFFFFFFF00000000, -int(s))
	if rv&uint64(mask) != 0 {
		return 0
	}
	if !isPPC64WordRotateMaskNonWrapping(mask) {
		return 0
	}
	return EncodePPC64RotateMask((32-s)&31, mask, 32)
}

// Test if a doubleword shift right feeding into a CLRLSLDI can be merged into RLWINM.
// Return the encoded RLWINM constant, or 0 if they cannot be merged.
func mergePPC64ClrlsldiSrd(sld, srd int64) int64 {
	mask_1 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(srd)
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(ssacore.GetPPC64Shiftmb(sld))

	// Rewrite mask to apply after the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(ssacore.GetPPC64Shiftsh(sld))

	r_1 := 64 - srd
	r_2 := ssacore.GetPPC64Shiftsh(sld)
	r_3 := (r_1 + r_2) & 63 // This can wrap.

	if uint64(uint32(mask_3)) != mask_3 || mask_3 == 0 {
		return 0
	}
	// This combine only works when selecting and shifting the lower 32 bits.
	v1 := bits.RotateLeft64(0xFFFFFFFF00000000, int(r_3))
	if v1&mask_3 != 0 {
		return 0
	}
	return EncodePPC64RotateMask(r_3&31, int64(mask_3), 32)
}

// Test if RLWINM opcode rlw clears the upper 32 bits of the
// result. Return rlw if it does, 0 otherwise.
func mergePPC64MovwzregRlwinm(rlw int64) int64 {
	_, mb, me, _ := ssacore.DecodePPC64RotateMask(rlw)
	if mb > me {
		return 0
	}
	return rlw
}

// Test if AND feeding into an ANDconst can be merged. Return the encoded RLWINM constant,
// or 0 if they cannot be merged.
func mergePPC64RlwinmAnd(rlw int64, mask uint32) int64 {
	r, _, _, mask_rlw := ssacore.DecodePPC64RotateMask(rlw)

	// Rotate the input mask, combine with the rlwnm mask, and test if it is still a valid rlwinm mask.
	r_mask := bits.RotateLeft32(mask, int(r))

	mask_out := (mask_rlw & uint64(r_mask))

	// Verify the result is still a valid bitmask of <= 32 bits.
	if !IsPPC64WordRotateMask(int64(mask_out)) {
		return 0
	}
	return EncodePPC64RotateMask(r, int64(mask_out), 32)
}

// Test if RLWINM feeding into SRDconst can be merged. Return the encoded RLIWNM constant,
// or 0 if they cannot be merged.
func mergePPC64SldiRlwinm(sldi, rlw int64) int64 {
	r_1, mb, me, mask_1 := ssacore.DecodePPC64RotateMask(rlw)
	if mb > me || mb < sldi {
		// Wrapping masks cannot be merged as the upper 32 bits are effectively undefined in this case.
		// Likewise, if mb is less than the shift amount, it cannot be merged.
		return 0
	}
	// combine the masks, and adjust for the final left shift.
	mask_3 := mask_1 << sldi
	r_3 := (r_1 + sldi) & 31 // This can wrap.

	// Verify the result is still a valid bitmask of <= 32 bits.
	if uint64(uint32(mask_3)) != mask_3 {
		return 0
	}
	return EncodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Convenience function to rotate a 32 bit constant value by another constant.
func rotateLeft32(v, rotate int64) int64 {
	return int64(bits.RotateLeft32(uint32(v), int(rotate)))
}
