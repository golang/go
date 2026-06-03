// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"math/bits"
	"testing"

	"cmd/internal/obj"
)

// decodeLogicalImmArrEncoding is translated from DecodeBitMasks in ARM64
// ASL. It implements the decoding logic of imm13 defined by ARM64 specification.
func decodeLogicalImmArrEncoding(imm13 uint32, arr uint32) (uint64, bool) {
	n := (imm13 >> 12) & 1
	immr := (imm13 >> 6) & 0x3F
	imms := imm13 & 0x3F

	notImms := (^imms) & 0x3F
	combined := (n << 6) | notImms

	// if immN::NOT(imms) == '000000x' then Undefined(); end;
	if combined <= 1 {
		return 0, false
	}

	lenVal := bits.Len32(uint32(combined)) - 1
	esize := uint64(1 << lenVal)

	levels := uint32((1 << lenVal) - 1)
	// if immediate && (imms AND levels) == levels then Undefined(); end;
	if (imms & levels) == levels {
		return 0, false
	}

	s := uint64(imms & levels)
	r := uint64(immr & levels)

	// welem has s + 1 ones
	welem := (uint64(1) << (s + 1)) - 1

	// Rotate right welem by r bits within esize bits window
	rotated := welem
	mask := (uint64(1) << esize) - 1
	if r > 0 {
		r %= esize
		rotated = ((welem >> r) | (welem << (esize - r))) & mask
	}

	// Replicate rotated to fill 64 bits (as encode replicates to 64 bits)
	wmask := uint64(0)
	for i := uint64(0); i < 64; i += esize {
		wmask |= rotated << i
	}

	// Now mask back to the arrangement's lane size M
	var M uint64
	switch arr {
	case ARNG_B:
		M = 8
	case ARNG_H:
		M = 16
	case ARNG_S:
		M = 32
	case ARNG_D:
		M = 64
	default:
		return 0, false
	}

	mask = (uint64(1) << M) - 1
	return wmask & mask, true
}

func FuzzLogicalImmArrEncoding(f *testing.F) {
	f.Add(uint64(0x55), uint32(ARNG_B))
	f.Add(uint64(0xAA), uint32(ARNG_B))
	f.Add(uint64(0x0F), uint32(ARNG_B))
	f.Add(uint64(0x0101), uint32(ARNG_H))
	f.Add(uint64(0x00FF), uint32(ARNG_H))
	f.Add(uint64(0x0000FFFF), uint32(ARNG_S))
	f.Add(uint64(0x55555555), uint32(ARNG_S))
	f.Add(uint64(0x0000FFFF), uint32(ARNG_D))
	f.Add(uint64(0x0101010101010101), uint32(ARNG_D))

	f.Fuzz(func(t *testing.T, v uint64, arr uint32) {
		if arr != ARNG_B && arr != ARNG_H && arr != ARNG_S && arr != ARNG_D {
			return
		}
		adjacentAddr := &obj.Addr{
			Type: obj.TYPE_REG,
			Reg:  int16(REG_ZARNG) + 0 + (int16(arr) << 5),
		}

		encoded, ok := encodeLogicalImmArrEncoding(v, adjacentAddr)
		if !ok {
			return
		}

		imm13 := encoded >> 5
		decoded, ok := decodeLogicalImmArrEncoding(imm13, arr)
		if !ok {
			t.Errorf("Failed to decode imm13=0x%x arr=%d", imm13, arr)
			return
		}

		var mask uint64
		switch arr {
		case ARNG_B:
			mask = 0xFF
		case ARNG_H:
			mask = 0xFFFF
		case ARNG_S:
			mask = 0xFFFFFFFF
		case ARNG_D:
			mask = 0xFFFFFFFFFFFFFFFF
		}
		expected := v & mask

		if decoded != expected {
			t.Errorf("Fuzz roundtrip failed for v=0x%x arr=%d. Expected 0x%x, got 0x%x (imm13=0x%x)", v, arr, expected, decoded, imm13)
		}
	})
}
