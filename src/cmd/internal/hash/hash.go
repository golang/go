// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hash implements hash functions used in the compiler toolchain.
package hash

import (
	"crypto/sha256"
	"hash"
)

// Size32 is the size of the 32-byte hash functions [New32] and [Sum32].
const Size32 = 32

// New32 returns a new [hash.Hash] computing the 32-byte hash checksum.
// Note that New32 and [Sum32] compute different hashes.
func New32() hash.Hash {
	h := sha256.New()
	_, _ = h.Write([]byte{1}) // make this hash different from sha256
	return h
}

// Sum32 returns a 32-byte checksum of the data.
// Note that Sum32 and [New32] compute different hashes.
func Sum32(data []byte) [32]byte {
	sum := sha256.Sum256(data)
	sum[0] ^= 0xff // make this hash different from sha256
	return sum
}
