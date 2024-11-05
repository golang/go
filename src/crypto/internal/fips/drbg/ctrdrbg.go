// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package drbg

import (
	"crypto/internal/fips"
	"crypto/internal/fips/aes"
	"crypto/internal/fips/subtle"
	"internal/byteorder"
	"math/bits"
)

// Counter is an SP 800-90A Rev. 1 CTR_DRBG instantiated with AES-256.
//
// Per Table 3, it has a security strength of 256 bits, a seed size of 384 bits,
// a counter length of 128 bits, a reseed interval of 2^48 requests, and a
// maximum request size of 2^19 bits (2^16 bytes, 64 KiB).
//
// We support a narrow range of parameters that fit the needs of our RNG:
// AES-256, no derivation function, no personalization string, no prediction
// resistance, and 384-bit additional input.
type Counter struct {
	// c is instantiated with K as the key and V as the counter.
	c aes.CTR

	reseedCounter uint64
}

const (
	keySize        = 256 / 8
	SeedSize       = keySize + aes.BlockSize
	reseedInterval = 1 << 48
	maxRequestSize = (1 << 19) / 8
)

func NewCounter(entropy *[SeedSize]byte) *Counter {
	// CTR_DRBG_Instantiate_algorithm, per Section 10.2.1.3.1.
	fips.RecordApproved()

	K := make([]byte, keySize)
	V := make([]byte, aes.BlockSize)

	// V starts at 0, but is incremented in CTR_DRBG_Update before each use,
	// unlike AES-CTR where it is incremented after each use.
	V[len(V)-1] = 1

	cipher, err := aes.New(K)
	if err != nil {
		panic(err)
	}

	c := &Counter{}
	c.c = *aes.NewCTR(cipher, V)
	c.update(entropy)
	c.reseedCounter = 1
	return c
}

func (c *Counter) update(seed *[SeedSize]byte) {
	// CTR_DRBG_Update, per Section 10.2.1.2.

	temp := make([]byte, SeedSize)
	c.c.XORKeyStream(temp, seed[:])
	K := temp[:keySize]
	V := temp[keySize:]

	// Again, we pre-increment V, like in NewCounter.
	increment((*[aes.BlockSize]byte)(V))

	cipher, err := aes.New(K)
	if err != nil {
		panic(err)
	}
	c.c = *aes.NewCTR(cipher, V)
}

func increment(v *[aes.BlockSize]byte) {
	hi := byteorder.BeUint64(v[:8])
	lo := byteorder.BeUint64(v[8:])
	lo, c := bits.Add64(lo, 1, 0)
	hi, _ = bits.Add64(hi, 0, c)
	byteorder.BePutUint64(v[:8], hi)
	byteorder.BePutUint64(v[8:], lo)
}

func (c *Counter) Reseed(entropy, additionalInput *[SeedSize]byte) {
	// CTR_DRBG_Reseed_algorithm, per Section 10.2.1.4.1.
	fips.RecordApproved()

	var seed [SeedSize]byte
	subtle.XORBytes(seed[:], entropy[:], additionalInput[:])
	c.update(&seed)
	c.reseedCounter = 1
}

// Generate produces at most maxRequestSize bytes of random data in out.
func (c *Counter) Generate(out []byte, additionalInput *[SeedSize]byte) (reseedRequired bool) {
	// CTR_DRBG_Generate_algorithm, per Section 10.2.1.5.1.
	fips.RecordApproved()

	if len(out) > maxRequestSize {
		panic("crypto/drbg: internal error: request size exceeds maximum")
	}

	// Step 1.
	if c.reseedCounter > reseedInterval {
		return true
	}

	// Step 2.
	if additionalInput != nil {
		c.update(additionalInput)
	} else {
		// If the additional input is null, the first CTR_DRBG_Update is
		// skipped, but the additional input is replaced with an all-zero string
		// for the second CTR_DRBG_Update.
		additionalInput = new([SeedSize]byte)
	}

	// Steps 3-5.
	clear(out)
	c.c.XORKeyStream(out, out)
	aes.RoundToBlock(&c.c)

	// Step 6.
	c.update(additionalInput)

	// Step 7.
	c.reseedCounter++

	// Step 8.
	return false
}
