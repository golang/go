// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package crc32

import (
	"internal/byteorder"
	"internal/cpu"
	"simd/archsimd"
)

// SHA3 selects cores whose vector units outrun their CRC32C loop. It is an
// empirical proxy, measured on Neoverse N1 (excluded), V1, V2 and V3.
var usePMULL = archsimd.ARM64.PMULL() && cpu.ARM64.HasSHA3

// (K(8k+63), K(8k-1)) for k = 128, 112, ..., 16 bytes,
// where K(e) = bitreverse32(x^e mod 0x11EDC6F41) << 32.
var castagnoliFoldK = [16]uint64{
	0x6577b24500000000, 0x7417153f00000000,
	0x169472b600000000, 0x963e61cd00000000,
	0xc92f998d00000000, 0x3365346a00000000,
	0x1c42da4300000000, 0x6d883e3800000000,
	0x1c19243b00000000, 0x75bba45b00000000,
	0xa46ef4aa00000000, 0x6051243f00000000,
	0x33ccbbbc00000000, 0xa2158b3400000000,
	0x3743f7bd00000000, 0x3171d43000000000,
}

// castagnoliUpdatePMULL computes castagnoliUpdate, folding inputs of at
// least pmullMinLength bytes with carry-less multiplies.
func castagnoliUpdatePMULL(crc uint32, p []byte) uint32 {
	// Tests call this directly with shorter inputs.
	if len(p) < pmullMinLength {
		return castagnoliUpdate(crc, p)
	}
	k := &castagnoliFoldK
	k128 := archsimd.LoadUint64x2(k[0:])
	x0 := load(p).Xor(archsimd.Uint64x2{}.SetElem(0, uint64(crc)))
	x1 := load(p[16:])
	x2 := load(p[32:])
	x3 := load(p[48:])
	x4 := load(p[64:])
	x5 := load(p[80:])
	x6 := load(p[96:])
	x7 := load(p[112:])
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
	// Move x0..x6 forward by 112..16 bytes onto x7.
	a := mul(x0, archsimd.LoadUint64x2(k[2:])).Xor(mul(x1, archsimd.LoadUint64x2(k[4:])))
	b := mul(x2, archsimd.LoadUint64x2(k[6:])).Xor(mul(x3, archsimd.LoadUint64x2(k[8:])))
	c := mul(x4, archsimd.LoadUint64x2(k[10:])).Xor(mul(x5, archsimd.LoadUint64x2(k[12:])))
	d := foldInto(x6, archsimd.LoadUint64x2(k[14:]), x7)
	r := a.Xor(b).Xor(c.Xor(d))

	var res [16]byte
	byteorder.LEPutUint64(res[:], r.GetElem(0))
	byteorder.LEPutUint64(res[8:], r.GetElem(1))
	// crc is already mixed into res.
	crc = castagnoliUpdate(0, res[:])
	if len(p) == 0 {
		return crc
	}
	return castagnoliUpdate(crc, p)
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
