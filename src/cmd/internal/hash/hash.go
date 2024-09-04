// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hash implements hash functions used in the compiler toolchain.
package hash

import (
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"hash"
)

const (
	// Size32 is the size of 32 bytes hash checksum.
	Size32 = sha256.Size
	// Size20 is the size of 20 bytes hash checksum.
	Size20 = sha1.Size
	// Size16 is the size of 16 bytes hash checksum.
	Size16 = md5.Size
)

// New32 returns a new [hash.Hash] computing the 32 bytes hash checksum.
func New32() hash.Hash {
	h := sha256.New()
	_, _ = h.Write([]byte{1}) // make this hash different from sha256
	return h
}

// New20 returns a new [hash.Hash] computing the 20 bytes hash checksum.
func New20() hash.Hash {
	return sha1.New()
}

// New16 returns a new [hash.Hash] computing the 16 bytes hash checksum.
func New16() hash.Hash {
	return md5.New()
}

// Sum32 returns the 32 bytes checksum of the data.
func Sum32(data []byte) [Size32]byte {
	sum := sha256.Sum256(data)
	sum[0] ^= 1 // make this hash different from sha256
	return sum
}

// Sum20 returns the 20 bytes checksum of the data.
func Sum20(data []byte) [Size20]byte {
	return sha1.Sum(data)
}

// Sum16 returns the 16 bytes checksum of the data.
func Sum16(data []byte) [Size16]byte {
	return md5.Sum(data)
}
