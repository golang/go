// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// This file contains helpers for the emulated narrowing integer conversions in
// converts_gen_amd64.go, which tmplgen generates. These are conversions that
// AVX-512 provides as single instructions but that we need without AVX-512:
//
//   - The 2:1 TruncTo and SaturateTo conversions on 128- and 256-bit vectors,
//     which only require AVX and AVX2, respectively.
//   - SaturateToUint from signed integers, which has no AVX-512 instruction,
//     at every size ratio.
//
// On AVX-512 hardware, a single VPMOV instruction does each 2:1 conversion, so
// the emulations use that instead where it saves instructions. (It doesn't for
// the 128-bit truncations, or for saturation that's a single pack.)
//
// TODO: With GOAMD64=v4, the feature checks could be resolved statically.

// The 256-bit emulations first narrow within each 128-bit group, which leaves
// the results in the low 64 bits of each group and zeros in the high 64 bits,
// and then use gatherLo64 to move the results of the upper group down.

var gatherLo64Idx = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// gatherLo64 returns the elements of x in the order 0, 2, 1, 3.
func gatherLo64(x Uint64x4) Uint64x4 {
	return x.ReshapeToUint32s().LookupMod(LoadUint32x8Array(&gatherLo64Idx)).ReshapeToUint64s()
}

// Byte permutations for truncation. -1 produces a zero.
var (
	trunc16To8Idx  = [16]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	trunc32To16Idx = [16]int8{0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1}

	trunc16To8GroupedIdx = [32]int8{
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	}
	trunc32To16GroupedIdx = [32]int8{
		0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
	}
)

// Saturation bounds.
var (
	maxUint8x8  = [8]uint16{maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8}
	maxUint16x4 = [4]uint32{maxUint16, maxUint16, maxUint16, maxUint16}
	maxInt32x2  = [2]int64{maxInt32, maxInt32}
	minInt32x2  = [2]int64{minInt32, minInt32}
	maxUint32x2 = [2]int64{maxUint32, maxUint32}
	maxUint8x16 = [16]uint16{maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8, maxUint8}
	maxUint16x8 = [8]uint32{maxUint16, maxUint16, maxUint16, maxUint16, maxUint16, maxUint16, maxUint16, maxUint16}
	maxInt32x4  = [4]int64{maxInt32, maxInt32, maxInt32, maxInt32}
	minInt32x4  = [4]int64{minInt32, minInt32, minInt32, minInt32}
	maxUint32x4 = [4]int64{maxUint32, maxUint32, maxUint32, maxUint32}
)
