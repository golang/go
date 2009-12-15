// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SHA1 block step.
// In its own file so that a faster assembly or C version
// can be substituted easily.

package sha1

const (
	_K0 = 0x5A827999
	_K1 = 0x6ED9EBA1
	_K2 = 0x8F1BBCDC
	_K3 = 0xCA62C1D6
)

func _Block(dig *digest, p []byte) int {
	var w [80]uint32

	n := 0
	h0, h1, h2, h3, h4 := dig.h[0], dig.h[1], dig.h[2], dig.h[3], dig.h[4]
	for len(p) >= _Chunk {
		// Can interlace the computation of w with the
		// rounds below if needed for speed.
		for i := 0; i < 16; i++ {
			j := i * 4
			w[i] = uint32(p[j])<<24 | uint32(p[j+1])<<16 | uint32(p[j+2])<<8 | uint32(p[j+3])
		}
		for i := 16; i < 80; i++ {
			tmp := w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]
			w[i] = tmp<<1 | tmp>>(32-1)
		}

		a, b, c, d, e := h0, h1, h2, h3, h4

		// Each of the four 20-iteration rounds
		// differs only in the computation of f and
		// the choice of K (_K0, _K1, etc).
		for i := 0; i < 20; i++ {
			f := b&c | (^b)&d
			a5 := a<<5 | a>>(32-5)
			b30 := b<<30 | b>>(32-30)
			t := a5 + f + e + w[i] + _K0
			a, b, c, d, e = t, a, b30, c, d
		}
		for i := 20; i < 40; i++ {
			f := b ^ c ^ d
			a5 := a<<5 | a>>(32-5)
			b30 := b<<30 | b>>(32-30)
			t := a5 + f + e + w[i] + _K1
			a, b, c, d, e = t, a, b30, c, d
		}
		for i := 40; i < 60; i++ {
			f := b&c | b&d | c&d
			a5 := a<<5 | a>>(32-5)
			b30 := b<<30 | b>>(32-30)
			t := a5 + f + e + w[i] + _K2
			a, b, c, d, e = t, a, b30, c, d
		}
		for i := 60; i < 80; i++ {
			f := b ^ c ^ d
			a5 := a<<5 | a>>(32-5)
			b30 := b<<30 | b>>(32-30)
			t := a5 + f + e + w[i] + _K3
			a, b, c, d, e = t, a, b30, c, d
		}

		h0 += a
		h1 += b
		h2 += c
		h3 += d
		h4 += e

		p = p[_Chunk:]
		n += _Chunk
	}

	dig.h[0], dig.h[1], dig.h[2], dig.h[3], dig.h[4] = h0, h1, h2, h3, h4
	return n
}
