// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

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

func NewRotateParams(start, end, amount int64) RotateParams {
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
		Start:  uint8(start),
		End:    uint8(end),
		Amount: uint8(amount),
	}
}
