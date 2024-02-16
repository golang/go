// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha1

import (
	"math/bits"
)

const (
	_K0 = 0x5A827999
	_K1 = 0x6ED9EBA1
	_K2 = 0x8F1BBCDC
	_K3 = 0xCA62C1D6
)

// blockGeneric is a portable, pure Go version of the SHA-1 block step.
// It's used by sha1block_generic.go and tests.
func blockGeneric(dig *digest, p []byte) {
	var w [16]uint32

	h0, h1, h2, h3, h4 := dig.h[0], dig.h[1], dig.h[2], dig.h[3], dig.h[4]
	for len(p) >= chunk {
		// Can interlace the computation of w with the
		// rounds below if needed for speed.
		for i := 0; i < 16; i++ {
			j := i * 4
			w[i] = uint32(p[j])<<24 | uint32(p[j+1])<<16 | uint32(p[j+2])<<8 | uint32(p[j+3])
		}

		a, b, c, d, e := h0, h1, h2, h3, h4

		// Each of the four 20-iteration rounds
		// differs only in the computation of f and
		// the choice of K (_K0, _K1, etc).
		i := 0
		for ; i < 16; i++ {
			f := b&c | (^b)&d
			t := bits.RotateLeft32(a, 5) + f + e + w[i&0xf] + _K0
			a, b, c, d, e = t, a, bits.RotateLeft32(b, 30), c, d
		}
		for ; i < 20; i++ {
			tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
			w[i&0xf] = bits.RotateLeft32(tmp, 1)

			f := b&c | (^b)&d
			t := bits.RotateLeft32(a, 5) + f + e + w[i&0xf] + _K0
			a, b, c, d, e = t, a, bits.RotateLeft32(b, 30), c, d
		}
		for ; i < 40; i++ {
			tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
			w[i&0xf] = bits.RotateLeft32(tmp, 1)
			f := b ^ c ^ d
			t := bits.RotateLeft32(a, 5) + f + e + w[i&0xf] + _K1
			a, b, c, d, e = t, a, bits.RotateLeft32(b, 30), c, d
		}
		for ; i < 60; i++ {
			tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
			w[i&0xf] = bits.RotateLeft32(tmp, 1)
			f := ((b | c) & d) | (b & c)
			t := bits.RotateLeft32(a, 5) + f + e + w[i&0xf] + _K2
			a, b, c, d, e = t, a, bits.RotateLeft32(b, 30), c, d
		}
		for ; i < 80; i++ {
			tmp := w[(i-3)&0xf] ^ w[(i-8)&0xf] ^ w[(i-14)&0xf] ^ w[(i)&0xf]
			w[i&0xf] = bits.RotateLeft32(tmp, 1)
			f := b ^ c ^ d
			t := bits.RotateLeft32(a, 5) + f + e + w[i&0xf] + _K3
			a, b, c, d, e = t, a, bits.RotateLeft32(b, 30), c, d
		}

		h0 += a
		h1 += b
		h2 += c
		h3 += d
		h4 += e

		p = p[chunk:]
	}

	dig.h[0], dig.h[1], dig.h[2], dig.h[3], dig.h[4] = h0, h1, h2, h3, h4
}
