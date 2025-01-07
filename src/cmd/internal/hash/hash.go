// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hash implements hash functions used in the compiler toolchain.
package hash

// TODO(rsc): Delete the 16 and 20 forms and use 32 at all call sites.

import (
	"crypto/sha256"
	"hash"
)

const (
	// Size32 is the size of the 32-byte hash checksum.
	Size32 = 32
	// Size20 is the size of the 20-byte hash checksum.
	Size20 = 20
	// Size16 is the size of the 16-byte hash checksum.
	Size16 = 16
)

type shortHash struct {
	hash.Hash
	n int
}

func (h *shortHash) Sum(b []byte) []byte {
	old := b
	sum := h.Hash.Sum(b)
	return sum[:len(old)+h.n]
}

// New32 returns a new [hash.Hash] computing the 32 bytes hash checksum.
func New32() hash.Hash {
	h := sha256.New()
	_, _ = h.Write([]byte{1}) // make this hash different from sha256
	return h
}

// New20 returns a new [hash.Hash] computing the 20 bytes hash checksum.
func New20() hash.Hash {
	return &shortHash{New32(), 20}
}

// New16 returns a new [hash.Hash] computing the 16 bytes hash checksum.
func New16() hash.Hash {
	return &shortHash{New32(), 16}
}

// Sum32 returns the 32 bytes checksum of the data.
func Sum32(data []byte) [Size32]byte {
	sum := sha256.Sum256(data)
	sum[0] ^= 1 // make this hash different from sha256
	return sum
}

// Sum20 returns the 20 bytes checksum of the data.
func Sum20(data []byte) [Size20]byte {
	sum := Sum32(data)
	var short [Size20]byte
	copy(short[:], sum[4:])
	return short
}

// Sum16 returns the 16 bytes checksum of the data.
func Sum16(data []byte) [Size16]byte {
	sum := Sum32(data)
	var short [Size16]byte
	copy(short[:], sum[8:])
	return short
}
