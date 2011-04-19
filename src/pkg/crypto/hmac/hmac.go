// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hmac implements the Keyed-Hash Message Authentication Code (HMAC) as
// defined in U.S. Federal Information Processing Standards Publication 198.
// An HMAC is a cryptographic hash that uses a key to sign a message.
// The receiver verifies the hash by recomputing it using the same key.
package hmac

import (
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"hash"
	"os"
)

// FIPS 198:
// http://csrc.nist.gov/publications/fips/fips198/fips-198a.pdf

// key is zero padded to 64 bytes
// ipad = 0x36 byte repeated to 64 bytes
// opad = 0x5c byte repeated to 64 bytes
// hmac = H([key ^ opad] H([key ^ ipad] text))

const (
	// NOTE(rsc): This constant is actually the
	// underlying hash function's block size.
	// HMAC is only conventionally used with
	// MD5 and SHA1, and both use 64-byte blocks.
	// The hash.Hash interface doesn't provide a
	// way to find out the block size.
	padSize = 64
)

type hmac struct {
	size         int
	key, tmp     []byte
	outer, inner hash.Hash
}

func (h *hmac) tmpPad(xor byte) {
	for i, k := range h.key {
		h.tmp[i] = xor ^ k
	}
	for i := len(h.key); i < padSize; i++ {
		h.tmp[i] = xor
	}
}

func (h *hmac) Sum() []byte {
	sum := h.inner.Sum()
	h.tmpPad(0x5c)
	for i, b := range sum {
		h.tmp[padSize+i] = b
	}
	h.outer.Reset()
	h.outer.Write(h.tmp)
	return h.outer.Sum()
}

func (h *hmac) Write(p []byte) (n int, err os.Error) {
	return h.inner.Write(p)
}

func (h *hmac) Size() int { return h.size }

func (h *hmac) Reset() {
	h.inner.Reset()
	h.tmpPad(0x36)
	h.inner.Write(h.tmp[0:padSize])
}

// New returns a new HMAC hash using the given hash generator and key.
func New(h func() hash.Hash, key []byte) hash.Hash {
	hm := new(hmac)
	hm.outer = h()
	hm.inner = h()
	hm.size = hm.inner.Size()
	hm.tmp = make([]byte, padSize+hm.size)
	if len(key) > padSize {
		// If key is too big, hash it.
		hm.outer.Write(key)
		key = hm.outer.Sum()
	}
	hm.key = make([]byte, len(key))
	copy(hm.key, key)
	hm.Reset()
	return hm
}

// NewMD5 returns a new HMAC-MD5 hash using the given key.
func NewMD5(key []byte) hash.Hash { return New(md5.New, key) }

// NewSHA1 returns a new HMAC-SHA1 hash using the given key.
func NewSHA1(key []byte) hash.Hash { return New(sha1.New, key) }

// NewSHA256 returns a new HMAC-SHA256 hash using the given key.
func NewSHA256(key []byte) hash.Hash { return New(sha256.New, key) }
