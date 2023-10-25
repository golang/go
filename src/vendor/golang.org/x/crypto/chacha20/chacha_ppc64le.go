// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc && !purego

package chacha20

const bufSize = 256

//go:noescape
func chaCha20_ctr32_vsx(out, inp *byte, len int, key *[8]uint32, counter *uint32)

func (c *Cipher) xorKeyStreamBlocks(dst, src []byte) {
	chaCha20_ctr32_vsx(&dst[0], &src[0], len(src), &c.key, &c.counter)
}
