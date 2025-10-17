// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package entropy

import "math/bits"

// This file includes a SHA-384 implementation to insulate the entropy source
// from any changes in the FIPS 140-3 module's crypto/internal/fips140/sha512
// package. We support 1024-byte inputs for the entropy source, and arbitrary
// length inputs for ACVP testing.

var initState = [8]uint64{
	0xcbbb9d5dc1059ed8,
	0x629a292a367cd507,
	0x9159015a3070dd17,
	0x152fecd8f70e5939,
	0x67332667ffc00b31,
	0x8eb44a8768581511,
	0xdb0c2e0d64f98fa7,
	0x47b5481dbefa4fa4,
}

func SHA384(p *[1024]byte) [48]byte {
	h := initState

	sha384Block(&h, (*[128]byte)(p[0:128]))
	sha384Block(&h, (*[128]byte)(p[128:256]))
	sha384Block(&h, (*[128]byte)(p[256:384]))
	sha384Block(&h, (*[128]byte)(p[384:512]))
	sha384Block(&h, (*[128]byte)(p[512:640]))
	sha384Block(&h, (*[128]byte)(p[640:768]))
	sha384Block(&h, (*[128]byte)(p[768:896]))
	sha384Block(&h, (*[128]byte)(p[896:1024]))

	var padlen [128]byte
	padlen[0] = 0x80
	bePutUint64(padlen[112+8:], 1024*8)
	sha384Block(&h, &padlen)

	return digestBytes(&h)
}

func TestingOnlySHA384(p []byte) [48]byte {
	if len(p) == 1024 {
		return SHA384((*[1024]byte)(p))
	}

	h := initState
	bitLen := uint64(len(p)) * 8

	// Process full 128-byte blocks.
	for len(p) >= 128 {
		sha384Block(&h, (*[128]byte)(p[:128]))
		p = p[128:]
	}

	// Process final block and padding.
	var finalBlock [128]byte
	copy(finalBlock[:], p)
	finalBlock[len(p)] = 0x80
	if len(p) >= 112 {
		sha384Block(&h, &finalBlock)
		finalBlock = [128]byte{}
	}
	bePutUint64(finalBlock[112+8:], bitLen)
	sha384Block(&h, &finalBlock)

	return digestBytes(&h)
}

func digestBytes(h *[8]uint64) [48]byte {
	var digest [48]byte
	bePutUint64(digest[0:], h[0])
	bePutUint64(digest[8:], h[1])
	bePutUint64(digest[16:], h[2])
	bePutUint64(digest[24:], h[3])
	bePutUint64(digest[32:], h[4])
	bePutUint64(digest[40:], h[5])
	return digest
}

var _K = [...]uint64{
	0x428a2f98d728ae22,
	0x7137449123ef65cd,
	0xb5c0fbcfec4d3b2f,
	0xe9b5dba58189dbbc,
	0x3956c25bf348b538,
	0x59f111f1b605d019,
	0x923f82a4af194f9b,
	0xab1c5ed5da6d8118,
	0xd807aa98a3030242,
	0x12835b0145706fbe,
	0x243185be4ee4b28c,
	0x550c7dc3d5ffb4e2,
	0x72be5d74f27b896f,
	0x80deb1fe3b1696b1,
	0x9bdc06a725c71235,
	0xc19bf174cf692694,
	0xe49b69c19ef14ad2,
	0xefbe4786384f25e3,
	0x0fc19dc68b8cd5b5,
	0x240ca1cc77ac9c65,
	0x2de92c6f592b0275,
	0x4a7484aa6ea6e483,
	0x5cb0a9dcbd41fbd4,
	0x76f988da831153b5,
	0x983e5152ee66dfab,
	0xa831c66d2db43210,
	0xb00327c898fb213f,
	0xbf597fc7beef0ee4,
	0xc6e00bf33da88fc2,
	0xd5a79147930aa725,
	0x06ca6351e003826f,
	0x142929670a0e6e70,
	0x27b70a8546d22ffc,
	0x2e1b21385c26c926,
	0x4d2c6dfc5ac42aed,
	0x53380d139d95b3df,
	0x650a73548baf63de,
	0x766a0abb3c77b2a8,
	0x81c2c92e47edaee6,
	0x92722c851482353b,
	0xa2bfe8a14cf10364,
	0xa81a664bbc423001,
	0xc24b8b70d0f89791,
	0xc76c51a30654be30,
	0xd192e819d6ef5218,
	0xd69906245565a910,
	0xf40e35855771202a,
	0x106aa07032bbd1b8,
	0x19a4c116b8d2d0c8,
	0x1e376c085141ab53,
	0x2748774cdf8eeb99,
	0x34b0bcb5e19b48a8,
	0x391c0cb3c5c95a63,
	0x4ed8aa4ae3418acb,
	0x5b9cca4f7763e373,
	0x682e6ff3d6b2b8a3,
	0x748f82ee5defb2fc,
	0x78a5636f43172f60,
	0x84c87814a1f0ab72,
	0x8cc702081a6439ec,
	0x90befffa23631e28,
	0xa4506cebde82bde9,
	0xbef9a3f7b2c67915,
	0xc67178f2e372532b,
	0xca273eceea26619c,
	0xd186b8c721c0c207,
	0xeada7dd6cde0eb1e,
	0xf57d4f7fee6ed178,
	0x06f067aa72176fba,
	0x0a637dc5a2c898a6,
	0x113f9804bef90dae,
	0x1b710b35131c471b,
	0x28db77f523047d84,
	0x32caab7b40c72493,
	0x3c9ebe0a15c9bebc,
	0x431d67c49c100d4c,
	0x4cc5d4becb3e42b6,
	0x597f299cfc657e2a,
	0x5fcb6fab3ad6faec,
	0x6c44198c4a475817,
}

func sha384Block(dh *[8]uint64, p *[128]byte) {
	var w [80]uint64
	for i := range 80 {
		if i < 16 {
			w[i] = beUint64(p[i*8:])
		} else {
			v1 := w[i-2]
			t1 := bits.RotateLeft64(v1, -19) ^ bits.RotateLeft64(v1, -61) ^ (v1 >> 6)
			v2 := w[i-15]
			t2 := bits.RotateLeft64(v2, -1) ^ bits.RotateLeft64(v2, -8) ^ (v2 >> 7)

			w[i] = t1 + w[i-7] + t2 + w[i-16]
		}
	}

	a, b, c, d, e, f, g, h := dh[0], dh[1], dh[2], dh[3], dh[4], dh[5], dh[6], dh[7]

	for i := range 80 {
		t1 := h + (bits.RotateLeft64(e, -14) ^ bits.RotateLeft64(e, -18) ^
			bits.RotateLeft64(e, -41)) + ((e & f) ^ (^e & g)) + _K[i] + w[i]
		t2 := (bits.RotateLeft64(a, -28) ^ bits.RotateLeft64(a, -34) ^
			bits.RotateLeft64(a, -39)) + ((a & b) ^ (a & c) ^ (b & c))

		h = g
		g = f
		f = e
		e = d + t1
		d = c
		c = b
		b = a
		a = t1 + t2
	}

	dh[0] += a
	dh[1] += b
	dh[2] += c
	dh[3] += d
	dh[4] += e
	dh[5] += f
	dh[6] += g
	dh[7] += h
}

func beUint64(b []byte) uint64 {
	_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
}

func bePutUint64(b []byte, v uint64) {
	_ = b[7] // early bounds check to guarantee safety of writes below
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
}
