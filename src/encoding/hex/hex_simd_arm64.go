// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package hex

import "simd/archsimd"

// haveSIMD is true because NEON is part of the arm64 baseline.
const haveSIMD = true

var hexDigits16 = [16]uint8{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'}

// encodeSIMD encodes the whole 16-byte blocks of src that fit in dst
// and returns the number of bytes of src it encoded.
func encodeSIMD(dst, src []byte) int {
	n := len(src)
	digits := archsimd.LoadUint8x16Array(&hexDigits16)
	lowNibble := archsimd.BroadcastUint8x16(0x0f)
	for len(src) >= 16 && len(dst) >= 32 {
		v := archsimd.LoadUint8x16Array((*[16]uint8)(src))
		hi, lo := digits.LookupOrZero(v.ShiftAllRight(4)), digits.LookupOrZero(v.And(lowNibble))
		hi.InterleaveLo(lo).StoreArray((*[16]uint8)(dst))
		hi.InterleaveHi(lo).StoreArray((*[16]uint8)(dst[16:]))
		src, dst = src[16:], dst[32:]
	}
	return n - len(src)
}

// decodeSIMD decodes the whole 32-character blocks of src that fit in
// dst, up to the first block holding a non-hex character, and returns
// the number of characters of src it decoded. It computes the nibbles
// as decodeSIMD does on amd64.
func decodeSIMD(dst, src []byte) int {
	n := len(src)
	c6 := archsimd.BroadcastUint8x16(0xc6)
	six := archsimd.BroadcastUint8x16(6)
	f0 := archsimd.BroadcastUint8x16(0xf0)
	upper := archsimd.BroadcastUint8x16(0xdf)
	bigA := archsimd.BroadcastUint8x16('A')
	ten := archsimd.BroadcastUint8x16(10)
	for len(src) >= 32 && len(dst) >= 16 {
		c1 := archsimd.LoadUint8x16Array((*[16]uint8)(src))
		c2 := archsimd.LoadUint8x16Array((*[16]uint8)(src[16:]))
		n1 := c1.Add(c6).SubSaturated(six).Sub(f0).Min(c1.And(upper).Sub(bigA).AddSaturated(ten))
		n2 := c2.Add(c6).SubSaturated(six).Sub(f0).Min(c2.And(upper).Sub(bigA).AddSaturated(ten))
		if n1.Max(n2).ReduceMax() > 15 {
			break
		}
		n1.ConcatEven(n2).ShiftAllLeft(4).Or(n1.ConcatOdd(n2)).StoreArray((*[16]uint8)(dst))
		src, dst = src[32:], dst[16:]
	}
	return n - len(src)
}
