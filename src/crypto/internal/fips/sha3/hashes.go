// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

// This file provides functions for creating instances of the SHA-3
// and SHAKE hash functions, as well as utility functions for hashing
// bytes.

import "crypto/internal/fips"

// New224 creates a new SHA3-224 hash.
// Its generic security strength is 224 bits against preimage attacks,
// and 112 bits against collision attacks.
func New224() fips.Hash {
	return new224()
}

// New256 creates a new SHA3-256 hash.
// Its generic security strength is 256 bits against preimage attacks,
// and 128 bits against collision attacks.
func New256() fips.Hash {
	return new256()
}

// New384 creates a new SHA3-384 hash.
// Its generic security strength is 384 bits against preimage attacks,
// and 192 bits against collision attacks.
func New384() fips.Hash {
	return new384()
}

// New512 creates a new SHA3-512 hash.
// Its generic security strength is 512 bits against preimage attacks,
// and 256 bits against collision attacks.
func New512() fips.Hash {
	return new512()
}

// TODO(fips): do this in the stdlib crypto/sha3 package.
//
//     crypto.RegisterHash(crypto.SHA3_224, New224)
//     crypto.RegisterHash(crypto.SHA3_256, New256)
//     crypto.RegisterHash(crypto.SHA3_384, New384)
//     crypto.RegisterHash(crypto.SHA3_512, New512)

const (
	dsbyteSHA3   = 0b00000110
	dsbyteKeccak = 0b00000001
	dsbyteShake  = 0b00011111
	dsbyteCShake = 0b00000100

	// rateK[c] is the rate in bytes for Keccak[c] where c is the capacity in
	// bits. Given the sponge size is 1600 bits, the rate is 1600 - c bits.
	rateK256  = (1600 - 256) / 8
	rateK448  = (1600 - 448) / 8
	rateK512  = (1600 - 512) / 8
	rateK768  = (1600 - 768) / 8
	rateK1024 = (1600 - 1024) / 8
)

func new224Generic() *state {
	return &state{rate: rateK448, outputLen: 28, dsbyte: dsbyteSHA3}
}

func new256Generic() *state {
	return &state{rate: rateK512, outputLen: 32, dsbyte: dsbyteSHA3}
}

func new384Generic() *state {
	return &state{rate: rateK768, outputLen: 48, dsbyte: dsbyteSHA3}
}

func new512Generic() *state {
	return &state{rate: rateK1024, outputLen: 64, dsbyte: dsbyteSHA3}
}

// NewLegacyKeccak256 creates a new Keccak-256 hash.
//
// Only use this function if you require compatibility with an existing cryptosystem
// that uses non-standard padding. All other users should use New256 instead.
func NewLegacyKeccak256() fips.Hash {
	return &state{rate: rateK512, outputLen: 32, dsbyte: dsbyteKeccak}
}

// NewLegacyKeccak512 creates a new Keccak-512 hash.
//
// Only use this function if you require compatibility with an existing cryptosystem
// that uses non-standard padding. All other users should use New512 instead.
func NewLegacyKeccak512() fips.Hash {
	return &state{rate: rateK1024, outputLen: 64, dsbyte: dsbyteKeccak}
}

// Sum224 returns the SHA3-224 digest of the data.
func Sum224(data []byte) (digest [28]byte) {
	h := New224()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum256 returns the SHA3-256 digest of the data.
func Sum256(data []byte) (digest [32]byte) {
	h := New256()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum384 returns the SHA3-384 digest of the data.
func Sum384(data []byte) (digest [48]byte) {
	h := New384()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum512 returns the SHA3-512 digest of the data.
func Sum512(data []byte) (digest [64]byte) {
	h := New512()
	h.Write(data)
	h.Sum(digest[:0])
	return
}
