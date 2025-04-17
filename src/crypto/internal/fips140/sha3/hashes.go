// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

// New224 returns a new Digest computing the SHA3-224 hash.
func New224() *Digest {
	return &Digest{rate: rateK448, outputLen: 28, dsbyte: dsbyteSHA3}
}

// New256 returns a new Digest computing the SHA3-256 hash.
func New256() *Digest {
	return &Digest{rate: rateK512, outputLen: 32, dsbyte: dsbyteSHA3}
}

// New384 returns a new Digest computing the SHA3-384 hash.
func New384() *Digest {
	return &Digest{rate: rateK768, outputLen: 48, dsbyte: dsbyteSHA3}
}

// New512 returns a new Digest computing the SHA3-512 hash.
func New512() *Digest {
	return &Digest{rate: rateK1024, outputLen: 64, dsbyte: dsbyteSHA3}
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

// NewLegacyKeccak256 returns a new Digest computing the legacy, non-standard
// Keccak-256 hash.
func NewLegacyKeccak256() *Digest {
	return &Digest{rate: rateK512, outputLen: 32, dsbyte: dsbyteKeccak}
}

// NewLegacyKeccak512 returns a new Digest computing the legacy, non-standard
// Keccak-512 hash.
func NewLegacyKeccak512() *Digest {
	return &Digest{rate: rateK1024, outputLen: 64, dsbyte: dsbyteKeccak}
}
