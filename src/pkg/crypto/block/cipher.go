// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The block package is deprecated, use cipher instead.
// The block package implements standard block cipher modes
// that can be wrapped around low-level block cipher implementations.
// See http://csrc.nist.gov/groups/ST/toolkit/BCM/current_modes.html
// and NIST Special Publication 800-38A.
package block

// A Cipher represents an implementation of block cipher
// using a given key.  It provides the capability to encrypt
// or decrypt individual blocks.  The mode implementations
// extend that capability to streams of blocks.
type Cipher interface {
	// BlockSize returns the cipher's block size.
	BlockSize() int

	// Encrypt encrypts the first block in src into dst.
	// Src and dst may point at the same memory.
	Encrypt(dst, src []byte)

	// Decrypt decrypts the first block in src into dst.
	// Src and dst may point at the same memory.
	Decrypt(dst, src []byte)
}

// Utility routines

func shift1(dst, src []byte) byte {
	var b byte
	for i := len(src) - 1; i >= 0; i-- {
		bb := src[i] >> 7
		dst[i] = src[i]<<1 | b
		b = bb
	}
	return b
}

func same(p, q []byte) bool {
	if len(p) != len(q) {
		return false
	}
	for i := 0; i < len(p); i++ {
		if p[i] != q[i] {
			return false
		}
	}
	return true
}

func dup(p []byte) []byte {
	q := make([]byte, len(p))
	copy(q, p)
	return q
}
