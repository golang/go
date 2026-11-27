// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package crc64

import (
	"internal/byteorder"
	"simd/archsimd"
)

// (K(8k+63), K(8k-1)) for k = 128, 64, 48, 32, 16 bytes,
// where K(e) = bitreverse64(x^e mod P).
var foldISO = [10]uint64{
	0xf501b0000001b000, 0xb001000000010000,
	0x01b001b1b0000001, 0xb100010100000001,
	0x76db6c7000000001, 0xe145150000000001,
	0x1b1ab00000000001, 0xa011000000000001,
	0x6b70000000000001, 0xf500000000000001,
}

var foldECMA = [10]uint64{
	0x8757d71d4fcc1000, 0xd7d86b2af73de740,
	0x6ae3efbb9dd441f3, 0x081f6054a7842df4,
	0xb5ea1af9c013aca4, 0x69a35d91c3730254,
	0x60095b008a9efa44, 0x3be653a30fe1af51,
	0xe05dd497ca393ae4, 0xdabe95afc7875f40,
}

// The fold works on whole 16-byte blocks.
const crc64FoldMinLength = 16

var usePMULL = archsimd.ARM64.PMULL()

// archFold folds a prefix of p for the tables MakeTable returns for ISO
// and ECMA, returning the updated crc and the unconsumed tail. It is kept
// inlinable so other tables and short inputs pay no call.
func archFold(crc uint64, tab *Table, p []byte) (uint64, []byte) {
	if len(p) >= crc64FoldMinLength {
		switch tab {
		case ecmaTable, isoTable:
			crc, p = foldPMULL(crc, tab, p)
		}
	}
	return crc, p
}

// foldPMULL folds only the tables returned by MakeTable(ISO) and MakeTable(ECMA).
func foldPMULL(crc uint64, tab *Table, p []byte) (uint64, []byte) {
	var helper *[8]Table
	var k *[10]uint64
	switch {
	case !usePMULL:
		return crc, p
	case tab == ecmaTable:
		helper, k = slicing8TableECMA, &foldECMA
	case tab == isoTable:
		helper, k = slicing8TableISO, &foldISO
	default:
		return crc, p
	}

	var res [16]byte
	n := len(p) &^ 15
	r := fold(crc, p[:n], k)
	byteorder.LEPutUint64(res[:], r.GetElem(0))
	byteorder.LEPutUint64(res[8:], r.GetElem(1))
	// crc is already mixed into res.
	crc, _ = slicingUpdate(0, helper, res[:])
	return slicingUpdate(crc, helper, p[n:])
}

// fold folds p (len a multiple of 16, >= 16) into a 16-byte residue,
// using eight accumulators across 128-byte blocks and one for the rest.
func fold(crc uint64, p []byte, k *[10]uint64) archsimd.Uint64x2 {
	k16 := archsimd.LoadUint64x2(k[8:])
	c := archsimd.BroadcastUint64x2(0).SetElem(0, crc)
	var x7 archsimd.Uint64x2
	if len(p) < 128 {
		x7 = load(p).Xor(c)
		p = p[16:]
	} else {
		k128 := archsimd.LoadUint64x2(k[0:])
		k64 := archsimd.LoadUint64x2(k[2:])
		x0 := load(p).Xor(c)
		x1 := load(p[16:])
		x2 := load(p[32:])
		x3 := load(p[48:])
		x4 := load(p[64:])
		x5 := load(p[80:])
		x6 := load(p[96:])
		x7 = load(p[112:])
		for p = p[128:]; len(p) >= 128; p = p[128:] {
			x0 = foldInto(x0, k128, load(p))
			x1 = foldInto(x1, k128, load(p[16:]))
			x2 = foldInto(x2, k128, load(p[32:]))
			x3 = foldInto(x3, k128, load(p[48:]))
			x4 = foldInto(x4, k128, load(p[64:]))
			x5 = foldInto(x5, k128, load(p[80:]))
			x6 = foldInto(x6, k128, load(p[96:]))
			x7 = foldInto(x7, k128, load(p[112:]))
		}
		x4 = foldInto(x0, k64, x4)
		x5 = foldInto(x1, k64, x5)
		x6 = foldInto(x2, k64, x6)
		x7 = foldInto(x3, k64, x7)
		// One more 64-byte round shortens the serial 16-byte loop below.
		if len(p) >= 64 {
			x4 = foldInto(x4, k64, load(p))
			x5 = foldInto(x5, k64, load(p[16:]))
			x6 = foldInto(x6, k64, load(p[32:]))
			x7 = foldInto(x7, k64, load(p[48:]))
			p = p[64:]
		}
		k48 := archsimd.LoadUint64x2(k[4:])
		k32 := archsimd.LoadUint64x2(k[6:])
		x7 = mul(x4, k48).Xor(mul(x5, k32)).Xor(foldInto(x6, k16, x7))
	}
	for ; len(p) >= 16; p = p[16:] {
		x7 = foldInto(x7, k16, load(p))
	}
	return x7
}

// mul moves x forward by the distance whose constants are k.
func mul(x, k archsimd.Uint64x2) archsimd.Uint64x2 {
	return x.CarrylessMultiplyEven(k).Xor(x.CarrylessMultiplyOdd(k))
}

// foldInto is mul(x, k).Xor(d), with d xored into the low product first to
// keep it off the loop-carried path.
func foldInto(x, k, d archsimd.Uint64x2) archsimd.Uint64x2 {
	return x.CarrylessMultiplyEven(k).Xor(d).Xor(x.CarrylessMultiplyOdd(k))
}

func load(p []byte) archsimd.Uint64x2 {
	return archsimd.LoadUint8x16(p).ReshapeToUint64s()
}

// slicingUpdate is update's slicing-by-8 loop, except that it also consumes
// a final 8-byte word, returning the < 8-byte tail. update keeps its inline
// copy: calling this from there cost Neoverse N1 about 7%.
func slicingUpdate(crc uint64, helperTable *[8]Table, p []byte) (uint64, []byte) {
	for len(p) >= 8 {
		crc ^= byteorder.LEUint64(p)
		crc = helperTable[7][crc&0xff] ^
			helperTable[6][(crc>>8)&0xff] ^
			helperTable[5][(crc>>16)&0xff] ^
			helperTable[4][(crc>>24)&0xff] ^
			helperTable[3][(crc>>32)&0xff] ^
			helperTable[2][(crc>>40)&0xff] ^
			helperTable[1][(crc>>48)&0xff] ^
			helperTable[0][crc>>56]
		p = p[8:]
	}
	return crc, p
}
