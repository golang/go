// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"math/bits"
)

// RotateParams represents the immediates required for a "rotate
// then ... selected bits instruction".
//
// The Start and End values are the indexes that represent
// the masked region. They are inclusive and are in big-
// endian order (bit 0 is the MSB, bit 63 is the LSB). They
// may wrap around.
//
// Some examples:
//
// Masked region             | Start | End
// --------------------------+-------+----
// 0x00_00_00_00_00_00_00_0f | 60    | 63
// 0xf0_00_00_00_00_00_00_00 | 0     | 3
// 0xf0_00_00_00_00_00_00_0f | 60    | 3
//
// The Amount value represents the amount to rotate the
// input left by. Note that this rotation is performed
// before the masked region is used.
type RotateParams struct {
	Start  uint8 // big-endian start bit index [0..63]
	End    uint8 // big-endian end bit index [0..63]
	Amount uint8 // amount to rotate left
}

// NewRotateParams creates a set of parameters representing a
// rotation left by the amount provided and a selection of the bits
// between the provided start and end indexes (inclusive).
//
// The start and end indexes and the rotation amount must all
// be in the range 0-63 inclusive or this function will panic.
func NewRotateParams(start, end, amount uint8) RotateParams {
	if start&^63 != 0 {
		panic("start out of bounds")
	}
	if end&^63 != 0 {
		panic("end out of bounds")
	}
	if amount&^63 != 0 {
		panic("amount out of bounds")
	}
	return RotateParams{
		Start:  start,
		End:    end,
		Amount: amount,
	}
}

// RotateLeft generates a new set of parameters with the rotation amount
// increased by the given value. The selected bits are left unchanged.
func (r RotateParams) RotateLeft(amount uint8) RotateParams {
	r.Amount += amount
	r.Amount &= 63
	return r
}

// OutMask provides a mask representing the selected bits.
func (r RotateParams) OutMask() uint64 {
	// Note: z must be unsigned for bootstrap compiler
	z := uint8(63-r.End+r.Start) & 63 // number of zero bits in mask
	return bits.RotateLeft64(^uint64(0)<<z, -int(r.Start))
}

// InMask provides a mask representing the selected bits relative
// to the source value (i.e. pre-rotation).
func (r RotateParams) InMask() uint64 {
	return bits.RotateLeft64(r.OutMask(), -int(r.Amount))
}

// OutMerge tries to generate a new set of parameters representing
// the intersection between the selected bits and the provided mask.
// If the intersection is unrepresentable (0 or not contiguous) nil
// will be returned.
func (r RotateParams) OutMerge(mask uint64) *RotateParams {
	mask &= r.OutMask()
	if mask == 0 {
		return nil
	}

	// normalize the mask so that the set bits are left aligned
	o := bits.LeadingZeros64(^mask)
	mask = bits.RotateLeft64(mask, o)
	z := bits.LeadingZeros64(mask)
	mask = bits.RotateLeft64(mask, z)

	// check that the normalized mask is contiguous
	l := bits.LeadingZeros64(^mask)
	if l+bits.TrailingZeros64(mask) != 64 {
		return nil
	}

	// update start and end positions (rotation amount remains the same)
	r.Start = uint8(o+z) & 63
	r.End = (r.Start + uint8(l) - 1) & 63
	return &r
}

// InMerge tries to generate a new set of parameters representing
// the intersection between the selected bits and the provided mask
// as applied to the source value (i.e. pre-rotation).
// If the intersection is unrepresentable (0 or not contiguous) nil
// will be returned.
func (r RotateParams) InMerge(mask uint64) *RotateParams {
	return r.OutMerge(bits.RotateLeft64(mask, int(r.Amount)))
}
