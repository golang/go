// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hmac implements the Keyed-Hash Message Authentication Code (HMAC) as
// defined in U.S. Federal Information Processing Standards Publication 198.
// An HMAC is a cryptographic hash that uses a key to sign a message.
// The receiver verifies the hash by recomputing it using the same key.
package hmac

import (
	"hash"
)

// FIPS 198:
// http://csrc.nist.gov/publications/fips/fips198/fips-198a.pdf

// key is zero padded to the block size of the hash function
// ipad = 0x36 byte repeated for key length
// opad = 0x5c byte repeated for key length
// hmac = H([key ^ opad] H([key ^ ipad] text))

type hmac struct {
	size         int
	blocksize    int
	key, tmp     []byte
	outer, inner hash.Hash
}

func (h *hmac) tmpPad(xor byte) {
	for i, k := range h.key {
		h.tmp[i] = xor ^ k
	}
	for i := len(h.key); i < h.blocksize; i++ {
		h.tmp[i] = xor
	}
}

func (h *hmac) Sum(in []byte) []byte {
	origLen := len(in)
	in = h.inner.Sum(in)
	h.tmpPad(0x5c)
	copy(h.tmp[h.blocksize:], in[origLen:])
	h.outer.Reset()
	h.outer.Write(h.tmp)
	return h.outer.Sum(in[:origLen])
}

func (h *hmac) Write(p []byte) (n int, err error) {
	return h.inner.Write(p)
}

func (h *hmac) Size() int { return h.size }

func (h *hmac) BlockSize() int { return h.blocksize }

func (h *hmac) Reset() {
	h.inner.Reset()
	h.tmpPad(0x36)
	h.inner.Write(h.tmp[0:h.blocksize])
}

// New returns a new HMAC hash using the given hash.Hash type and key.
func New(h func() hash.Hash, key []byte) hash.Hash {
	hm := new(hmac)
	hm.outer = h()
	hm.inner = h()
	hm.size = hm.inner.Size()
	hm.blocksize = hm.inner.BlockSize()
	hm.tmp = make([]byte, hm.blocksize+hm.size)
	if len(key) > hm.blocksize {
		// If key is too big, hash it.
		hm.outer.Write(key)
		key = hm.outer.Sum(nil)
	}
	hm.key = make([]byte, len(key))
	copy(hm.key, key)
	hm.Reset()
	return hm
}
