// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package adler32

import "simd/archsimd"

// haveSIMD reports whether the CPU supports the AVX2 instructions
// used by updateSIMD.
var haveSIMD = archsimd.X86.AVX2()

const (
	// minSIMD is the smallest input length for which updateSIMD
	// outperforms updateGeneric.
	minSIMD = 64

	// blockSize is the number of bytes processed per iteration of
	// the vector loop.
	blockSize = 32

	// nmaxSIMD is nmax rounded down to a multiple of blockSize. The
	// vector loop processes at most this many bytes between modular
	// reductions.
	nmaxSIMD = nmax - nmax%blockSize
)

// taps[i] is the number of times the i'th byte of a block is counted
// in the second sum: blockSize times for the first byte down to once
// for the last. Cross-block contributions are accounted for by vps
// in updateSIMD.
var taps = [blockSize]int8{
	32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
	16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
}

// updateSIMD computes the checksum using AVX2 vector instructions,
// processing blockSize bytes per iteration. It accumulates, in
// uint32 lanes:
//
//	vs1: the plain sum of all bytes, via VPSADBW against zero,
//	vs2: the taps-weighted sum of the bytes of each block, via
//	     VPMADDUBSW with the taps and VPMADDWD with 1,
//	vps: the sum, over all blocks, of vs1 as it stood before that block,
//
// so that after an n-byte run starting from state (s1, s2),
// s1' = s1 + sum(vs1) and
// s2' = s2 + n*s1 + blockSize*sum(vps) + sum(vs2).
//
// The VPMADDUBSW pair sums are at most 255*(32+31) < 2^15-1, so they
// cannot saturate. Modular reduction is deferred to the end of each
// run. A run is at most nmaxSIMD <= nmax bytes, so by nmax's
// defining property the total contribution to s2', and hence every
// uint32 lane, stays below 2^32.
func updateSIMD(d digest, p []byte) digest {
	s1, s2 := uint32(d&0xffff), uint32(d>>16)

	w := archsimd.LoadInt8x32Array(&taps)
	ones := archsimd.BroadcastInt16x16(1)
	var zero archsimd.Uint8x32

	for len(p) >= blockSize {
		n := nmaxSIMD
		if n > len(p) {
			n = len(p) - len(p)%blockSize
		}
		q := p[:n]
		var vs1, vs2, vps archsimd.Uint32x8
		for len(q) >= blockSize {
			b := archsimd.LoadUint8x32(q)
			vps = vps.Add(vs1)
			vs1 = vs1.Add(b.SumOf8AbsDiff(zero).AsUint32x8())
			vs2 = vs2.Add(b.DotProductPairsSaturated(w).DotProductPairs(ones).AsUint32x8())
			q = q[blockSize:]
		}
		vs2 = vs2.Add(vps.ShiftAllLeft(5)) // 32 = blockSize bytes per block
		r1 := vs1.GetLo().Add(vs1.GetHi())
		r2 := vs2.GetLo().Add(vs2.GetHi())
		s2 += uint32(n)*s1 + r2.GetElem(0) + r2.GetElem(1) + r2.GetElem(2) + r2.GetElem(3)
		s1 += r1.GetElem(0) + r1.GetElem(1) + r1.GetElem(2) + r1.GetElem(3)
		s1 %= mod
		s2 %= mod
		p = p[n:]
	}
	if len(p) > 0 {
		return updateGeneric(digest(s2<<16|s1), p)
	}
	return digest(s2<<16 | s1)
}
