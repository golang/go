// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package adler32

import "simd/archsimd"

const (
	haveSIMD = true

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
var taps = [blockSize]uint16{
	32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
	16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
}

// updateSIMD computes the checksum using NEON vector instructions,
// processing blockSize bytes per iteration. It accumulates, in
// uint32 lanes:
//
//	vs1: the plain sum of all bytes,
//	vs2: the taps-weighted sum of the bytes of each block,
//	vps: the sum, over all blocks, of vs1 as it stood before that block,
//
// so that after an n-byte run starting from state (s1, s2),
// s1' = s1 + sum(vs1) and
// s2' = s2 + n*s1 + blockSize*sum(vps) + sum(vs2).
//
// Modular reduction is deferred to the end of each run. A run is at
// most nmaxSIMD <= nmax bytes, so by nmax's defining property the
// total contribution to s2', and hence every uint32 lane, stays
// below 2^32.
func updateSIMD(d digest, p []byte) digest {
	s1, s2 := uint32(d&0xffff), uint32(d>>16)

	w0 := archsimd.LoadUint16x8(taps[0:])
	w1 := archsimd.LoadUint16x8(taps[8:])
	w2 := archsimd.LoadUint16x8(taps[16:])
	w3 := archsimd.LoadUint16x8(taps[24:])

	for len(p) >= blockSize {
		n := nmaxSIMD
		if n > len(p) {
			n = len(p) - len(p)%blockSize
		}
		q := p[:n]
		var vs1, vs2, vps archsimd.Uint32x4
		for len(q) >= blockSize {
			a := archsimd.LoadUint8x16(q)
			b := archsimd.LoadUint8x16(q[16:])
			vps = vps.Add(vs1)

			// Widen the bytes to uint16 lanes.
			alo := a.ExtendLo8ToUint16()
			ahi := a.HiToLo().ExtendLo8ToUint16()
			blo := b.ExtendLo8ToUint16()
			bhi := b.HiToLo().ExtendLo8ToUint16()

			// Byte sums for vs1: each lane of t sums 4 bytes
			// (<= 1020) and of t2 8 bytes (<= 2040), so uint16
			// cannot overflow. The low half of t2 covers all of t.
			t := alo.Add(ahi).Add(blo.Add(bhi))
			t2 := t.ConcatAddPairs(t)
			vs1 = vs1.Add(t2.ExtendLo4ToUint32())

			// Weighted sums for vs2: each product is at most 255*32
			// = 8160, each lane of r sums 4 products (<= 32640) and
			// of r2 8 (<= 65280), so uint16 cannot overflow. The
			// low half of r2 covers all 32 products.
			pa := alo.Mul(w0)
			pb := ahi.Mul(w1)
			pc := blo.Mul(w2)
			pd := bhi.Mul(w3)
			r := pa.ConcatAddPairs(pb).ConcatAddPairs(pc.ConcatAddPairs(pd))
			r2 := r.ConcatAddPairs(r)
			vs2 = vs2.Add(r2.ExtendLo4ToUint32())

			q = q[blockSize:]
		}
		vs2 = vs2.Add(vps.ShiftAllLeft(5)) // 32 = blockSize bytes per block
		s2 += uint32(n)*s1 + vs2.ReduceSum()
		s1 += vs1.ReduceSum()
		s1 %= mod
		s2 %= mod
		p = p[n:]
	}
	if len(p) > 0 {
		return updateGeneric(digest(s2<<16|s1), p)
	}
	return digest(s2<<16 | s1)
}
