// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32 arm,!nacl 386

package rc4

func xorKeyStream(dst, src *byte, n int, state *[256]uint32, i, j *uint8)

// XORKeyStream sets dst to the result of XORing src with the key stream.
// Dst and src may be the same slice but otherwise should not overlap.
func (c *Cipher) XORKeyStream(dst, src []byte) {
	if len(src) == 0 {
		return
	}
	xorKeyStream(&dst[0], &src[0], len(src), &c.s, &c.i, &c.j)
}
