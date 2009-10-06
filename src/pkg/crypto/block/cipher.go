// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The block package implements standard block cipher modes
// that can be wrapped around low-level block cipher implementations.
// See http://csrc.nist.gov/groups/ST/toolkit/BCM/current_modes.html
// and NIST Special Publication 800-38A.
package block

import "io"

// A Cipher represents an implementation of block cipher
// using a given key.  It provides the capability to encrypt
// or decrypt individual blocks.  The mode implementations
// extend that capability to streams of blocks.
type Cipher interface {
	// BlockSize returns the cipher's block size.
	BlockSize() int;

	// Encrypt encrypts the first block in src into dst.
	// Src and dst may point at the same memory.
	Encrypt(src, dst []byte);

	// Decrypt decrypts the first block in src into dst.
	// Src and dst may point at the same memory.
	Decrypt(src, dst []byte);
}

// TODO(rsc): Digest belongs elsewhere.

// A Digest is an implementation of a message digest algorithm.
// Write data to it and then call Sum to retreive the digest.
// Calling Reset resets the internal state, as though no data has
// been written.
type Digest interface {
	io.Writer;
	Sum() []byte;
	Reset();
}


// Utility routines

func shift1(src, dst []byte) byte {
	var b byte;
	for i := len(src)-1; i >= 0; i-- {
		bb := src[i]>>7;
		dst[i] = src[i]<<1 | b;
		b = bb;
	}
	return b;
}

func same(p, q []byte) bool {
	if len(p) != len(q) {
		return false;
	}
	for i := 0; i < len(p); i++ {
		if p[i] != q[i] {
			return false;
		}
	}
	return true;
}

func copy(p []byte) []byte {
	q := make([]byte, len(p));
	for i, b := range p {
		q[i] = b;
	}
	return q;
}
