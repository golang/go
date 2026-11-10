// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package hex

import (
	"internal/byteorder"
	"simd"
)

// encodePortable and decodePortable are encodeSIMD and decodeSIMD written
// with the portable simd package. Only tests and benchmarks use them, to
// compare the two.

// encodePortable encodes the whole vectors of src that fit in dst and
// returns the number of bytes of src it encoded. The simd package cannot
// move bytes between lanes, so each 64-bit lane computes the digits of its
// own 8 bytes, and scalar stores put the two halves in order.
func encodePortable(dst, src []byte) int {
	vl := simd.VectorBitSize() / 8
	if simd.Emulated() || len(src) < vl {
		return 0
	}
	n := len(src)
	var lo, hi [8]uint64
	low32 := simd.BroadcastUint64s(0xffffffff)
	m16 := simd.BroadcastUint64s(0x0000ffff0000ffff)
	m8 := simd.BroadcastUint64s(0x00ff00ff00ff00ff)
	nibble := simd.BroadcastUint64s(0x000f000f000f000f)
	six := simd.BroadcastUint64s(0x0606060606060606)
	ones := simd.BroadcastUint64s(0x0101010101010101)
	zero := simd.BroadcastUint64s(0x3030303030303030)
	gap := simd.BroadcastUint64s(0x2727272727272727)
	// digits spreads the 4 bytes in the low half of each lane to 16-bit
	// slots, puts their nibbles in output order and adds '0', or 'a'-10
	// to the nibbles above 9.
	digits := func(x simd.Uint64s) simd.Uint64s {
		x = x.Or(x.ShiftAllLeft(16)).And(m16)
		x = x.Or(x.ShiftAllLeft(8)).And(m8)
		n := x.ShiftAllRight(4).And(nibble).Or(x.And(nibble).ShiftAllLeft(8))
		m := n.Add(six).ShiftAllRight(4).And(ones)
		return n.Add(zero).Add(m.ShiftAllLeft(8).Sub(m).And(gap))
	}
	for len(src) >= vl && len(dst) >= 2*vl {
		x := simd.LoadUint8s(src).ReshapeToUint64s()
		digits(x.And(low32)).Store(lo[:])
		digits(x.ShiftAllRight(32)).Store(hi[:])
		for k := range vl / 8 {
			byteorder.LEPutUint64(dst[16*k:], lo[k])
			byteorder.LEPutUint64(dst[16*k+8:], hi[k])
		}
		src, dst = src[vl:], dst[2*vl:]
	}
	return n - len(src)
}

// decodePortable decodes the whole vectors of src that fit in dst, up to
// the first vector holding a non-hex character, and returns the number of
// characters of src it decoded. It computes the nibbles as decodeSIMD does
// and packs each 64-bit lane in place; the simd package cannot narrow, so
// scalar stores join the 4-byte halves.
func decodePortable(dst, src []byte) int {
	vl := simd.VectorBitSize() / 8
	if simd.Emulated() || len(src) < vl {
		return 0
	}
	n := len(src)
	var out [8]uint64
	c6 := simd.BroadcastUint8s(0xc6)
	six := simd.BroadcastUint8s(6)
	f0 := simd.BroadcastUint8s(0xf0)
	upper := simd.BroadcastUint8s(0xdf)
	bigA := simd.BroadcastUint8s('A')
	ten := simd.BroadcastUint8s(10)
	fifteen := simd.BroadcastUint8s(15)
	m8 := simd.BroadcastUint64s(0x00ff00ff00ff00ff)
	m16 := simd.BroadcastUint64s(0x0000ffff0000ffff)
	for len(src) >= vl && len(dst) >= vl/2 {
		c := simd.LoadUint8s(src)
		nib := c.Add(c6).SubSaturated(six).Sub(f0).Min(c.And(upper).Sub(bigA).AddSaturated(ten))
		if !nib.Max(fifteen).Equal(fifteen).All() {
			break
		}
		v := nib.ReshapeToUint64s()
		v = v.ShiftAllLeft(4).Or(v.ShiftAllRight(8)).And(m8)
		v = v.Or(v.ShiftAllRight(8)).And(m16)
		v.Or(v.ShiftAllRight(16)).Store(out[:])
		for k := range vl / 16 {
			byteorder.LEPutUint64(dst[8*k:], out[2*k]&0xffffffff|out[2*k+1]<<32)
		}
		src, dst = src[vl:], dst[vl/2:]
	}
	return n - len(src)
}
