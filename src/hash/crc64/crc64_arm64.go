// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc64

import (
	"internal/byteorder"
	"internal/cpu"
)

// crc64FoldPMULL folds p (len a multiple of 16, >= 16) into a 16-byte residue.
//
//go:noescape
func crc64FoldPMULL(crc uint64, p []byte, k *[10]uint64, res *[16]byte)

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

var usePMULL = cpu.ARM64.HasPMULL

// archFold is kept inlinable so other tables and short inputs pay no call.
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
	crc64FoldPMULL(crc, p[:n], k, &res)
	// crc is already mixed into res.
	crc, _ = slicingUpdate(0, helper, res[:])
	return slicingUpdate(crc, helper, p[n:])
}

// slicingUpdate is update's slicing-by-8 loop, returning the < 8-byte tail.
// update keeps its inline copy: calling this from there cost Neoverse N1 about 7%.
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
