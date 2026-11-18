// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

package archsimd

// This file contains constant permutation index tables for the emulated
// shuffles in shuffles_gen_amd64.go, which tmplgen generates.
//
// The 128- and 256-bit emulations also reuse gatherLo64, gatherLo64Idx,
// trunc16To8Idx, trunc32To16Idx, trunc16To8GroupedIdx, and
// trunc32To16GroupedIdx from converts_emulated_amd64.go.

// 512-bit InterleaveLo and InterleaveHi index vectors for ConcatLookupMod.
var (
	interleaveLo64x8Idx = [8]uint64{0, 8, 1, 9, 2, 10, 3, 11}
	interleaveHi64x8Idx = [8]uint64{4, 12, 5, 13, 6, 14, 7, 15}

	interleaveLo32x16Idx = [16]uint32{
		0, 16, 1, 17, 2, 18, 3, 19,
		4, 20, 5, 21, 6, 22, 7, 23,
	}
	interleaveHi32x16Idx = [16]uint32{
		8, 24, 9, 25, 10, 26, 11, 27,
		12, 28, 13, 29, 14, 30, 15, 31,
	}

	interleaveLo16x32Idx = [32]uint16{
		0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
		8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
	}
	interleaveHi16x32Idx = [32]uint16{
		16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
		24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63,
	}

	interleaveLo8x64Idx = [64]uint8{
		0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71,
		8, 72, 9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
		16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
		24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95,
	}
	interleaveHi8x64Idx = [64]uint8{
		32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103,
		40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111,
		48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119,
		56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127,
	}

	// 128-bit lane interleaving of two 512-bit vectors via Uint64x8.ConcatLookupMod,
	// used for baseline AVX-512 (non-VBMI) 8-bit InterleaveLo and InterleaveHi.
	interleaveLo128x4Idx = [8]uint64{0, 1, 8, 9, 2, 3, 10, 11}
	interleaveHi128x4Idx = [8]uint64{4, 5, 12, 13, 6, 7, 14, 15}
)

// ConcatEven and ConcatOdd index vectors for ConcatLookupMod and LookupModOrZeroGrouped.
var (
	concatEven16x16Idx = [16]uint16{
		0, 2, 4, 6, 8, 10, 12, 14,
		16, 18, 20, 22, 24, 26, 28, 30,
	}
	concatOdd16x16Idx = [16]uint16{
		1, 3, 5, 7, 9, 11, 13, 15,
		17, 19, 21, 23, 25, 27, 29, 31,
	}

	concatEven8x32Idx = [32]uint8{
		0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
		32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
	}
	concatOdd8x32Idx = [32]uint8{
		1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
		33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
	}

	concatEven64x8Idx = [8]uint64{0, 2, 4, 6, 8, 10, 12, 14}
	concatOdd64x8Idx  = [8]uint64{1, 3, 5, 7, 9, 11, 13, 15}

	concatEven32x16Idx = [16]uint32{
		0, 2, 4, 6, 8, 10, 12, 14,
		16, 18, 20, 22, 24, 26, 28, 30,
	}
	concatOdd32x16Idx = [16]uint32{
		1, 3, 5, 7, 9, 11, 13, 15,
		17, 19, 21, 23, 25, 27, 29, 31,
	}

	concatEven16x32Idx = [32]uint16{
		0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
		32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
	}
	concatOdd16x32Idx = [32]uint16{
		1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
		33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
	}

	concatEven8x64Idx = [64]uint8{
		0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
		32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
		64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
		96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
	}
	concatOdd8x64Idx = [64]uint8{
		1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
		33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
		65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95,
		97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127,
	}

	// Per-128-bit-lane byte selectors packing even/odd bytes into the low 64 bits
	// of each lane, used for baseline AVX-512 (non-VBMI) 8-bit ConcatEven and ConcatOdd.
	concatEven8x64GroupedIdx = [64]int8{
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	}
	concatOdd8x64GroupedIdx = [64]int8{
		1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
		1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
		1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
		1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
	}
)
