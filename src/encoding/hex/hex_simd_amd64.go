// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package hex

import "simd/archsimd"

// haveSIMD reports whether the CPU supports the AVX2 instructions
// used by encodeSIMD and decodeSIMD.
var haveSIMD = archsimd.X86.AVX2()

var hexDigits32 = [32]uint8{
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
}

// encodeSIMD encodes the whole 16-byte blocks of src that fit in dst
// and returns the number of bytes of src it encoded.
func encodeSIMD(dst, src []byte) int {
	n := len(src)
	digits := archsimd.LoadUint8x32Array(&hexDigits32)
	spread := archsimd.BroadcastUint16x16(0x1001)
	for len(src) >= 32 && len(dst) >= 64 {
		encodeBlock(digits, spread, (*[16]uint8)(src), (*[32]uint8)(dst))
		encodeBlock(digits, spread, (*[16]uint8)(src[16:]), (*[32]uint8)(dst[32:]))
		src, dst = src[32:], dst[64:]
	}
	if len(src) >= 16 && len(dst) >= 32 {
		encodeBlock(digits, spread, (*[16]uint8)(src), (*[32]uint8)(dst))
		src = src[16:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(src)
}

// encodeBlock widens each byte 0xHL of src to the uint16 0x00HL.
// Multiplied by 0x1001 that is 0xL0HL, and shifted right by 4 it is
// 0x0L0H: the two nibbles in output order, which one in-lane byte
// shuffle turns into digits.
func encodeBlock(digits archsimd.Uint8x32, spread archsimd.Uint16x16, src *[16]uint8, dst *[32]uint8) {
	w := archsimd.LoadUint8x16Array(src).ExtendToUint16().Mul(spread).ShiftAllRight(4)
	digits.PermuteOrZeroGrouped(w.AsUint8x32().AsInt8x32()).StoreArray(dst)
}

// pairWeights makes VPMADDUBSW compute 16*first + second for each pair
// of nibbles.
var pairWeights = [32]int8{
	16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
	16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
}

// packBytes moves the low byte of each uint16 of the low 128-bit lane
// to bytes 0-7 and of the high lane to bytes 8-15.
var packBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14,
}

// decodeSIMD decodes the whole 32-character blocks of src that fit in
// dst, up to the first block holding a non-hex character, and returns
// the number of characters of src it decoded.
//
// It uses algorithm 3 of
// http://0x80.pl/notesen/2022-01-17-validating-hex-parse.html:
// a digit maps to 0-9 and a letter of either case to 10-15, anything
// else to more than 15 on both paths, so the smaller of the two is the
// nibble.
func decodeSIMD(dst, src []byte) int {
	n := len(src)
	c6 := archsimd.BroadcastUint8x32(0xc6)
	six := archsimd.BroadcastUint8x32(6)
	f0 := archsimd.BroadcastUint8x32(0xf0)
	upper := archsimd.BroadcastUint8x32(0xdf)
	bigA := archsimd.BroadcastUint8x32('A')
	ten := archsimd.BroadcastUint8x32(10)
	weights := archsimd.LoadInt8x32Array(&pairWeights)
	pack := archsimd.LoadInt8x32Array(&packBytes)
	for len(src) >= 32 && len(dst) >= 16 {
		c := archsimd.LoadUint8x32Array((*[32]uint8)(src))
		nib := c.Add(c6).SubSaturated(six).Sub(f0).Min(c.And(upper).Sub(bigA).AddSaturated(ten))
		if !nib.And(f0).IsZero() {
			break
		}
		b := nib.DotProductPairsSaturated(weights).AsUint8x32().PermuteOrZeroGrouped(pack)
		b.GetLo().Or(b.GetHi()).StoreArray((*[16]uint8)(dst))
		src, dst = src[32:], dst[16:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(src)
}
