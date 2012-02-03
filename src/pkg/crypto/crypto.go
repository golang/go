// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crypto collects common cryptographic constants.
package crypto

import (
	"hash"
)

// Hash identifies a cryptographic hash function that is implemented in another
// package.
type Hash uint

const (
	MD4       Hash = 1 + iota // import code.google.com/p/go.crypto/md4
	MD5                       // import crypto/md5
	SHA1                      // import crypto/sha1
	SHA224                    // import crypto/sha256
	SHA256                    // import crypto/sha256
	SHA384                    // import crypto/sha512
	SHA512                    // import crypto/sha512
	MD5SHA1                   // no implementation; MD5+SHA1 used for TLS RSA
	RIPEMD160                 // import code.google.com/p/go.crypto/ripemd160
	maxHash
)

var digestSizes = []uint8{
	MD4:       16,
	MD5:       16,
	SHA1:      20,
	SHA224:    28,
	SHA256:    32,
	SHA384:    48,
	SHA512:    64,
	MD5SHA1:   36,
	RIPEMD160: 20,
}

// Size returns the length, in bytes, of a digest resulting from the given hash
// function. It doesn't require that the hash function in question be linked
// into the program.
func (h Hash) Size() int {
	if h > 0 && h < maxHash {
		return int(digestSizes[h])
	}
	panic("crypto: Size of unknown hash function")
}

var hashes = make([]func() hash.Hash, maxHash)

// New returns a new hash.Hash calculating the given hash function. New panics
// if the hash function is not linked into the binary.
func (h Hash) New() hash.Hash {
	if h > 0 && h < maxHash {
		f := hashes[h]
		if f != nil {
			return f()
		}
	}
	panic("crypto: requested hash function is unavailable")
}

// Available reports whether the given hash function is linked into the binary.
func (h Hash) Available() bool {
	return h < maxHash && hashes[h] != nil
}

// RegisterHash registers a function that returns a new instance of the given
// hash function. This is intended to be called from the init function in
// packages that implement hash functions.
func RegisterHash(h Hash, f func() hash.Hash) {
	if h >= maxHash {
		panic("crypto: RegisterHash of unknown hash function")
	}
	hashes[h] = f
}

// PrivateKey represents a private key using an unspecified algorithm.
type PrivateKey interface{}
