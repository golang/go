// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/subtle"
)

// CMAC implements the CMAC mode from NIST SP 800-38B.
//
// It is optimized for use in Counter KDF (SP 800-108r1) and XAES-256-GCM
// (https://c2sp.org/XAES-256-GCM), rather than for exposing it to applications
// as a stand-alone MAC.
type CMAC struct {
	b  aes.Block
	k1 [aes.BlockSize]byte
	k2 [aes.BlockSize]byte
}

func NewCMAC(b *aes.Block) *CMAC {
	c := &CMAC{b: *b}
	c.deriveSubkeys()
	return c
}

func (c *CMAC) deriveSubkeys() {
	c.b.Encrypt(c.k1[:], c.k1[:])
	msb := shiftLeft(&c.k1)
	c.k1[len(c.k1)-1] ^= msb * 0b10000111

	c.k2 = c.k1
	msb = shiftLeft(&c.k2)
	c.k2[len(c.k2)-1] ^= msb * 0b10000111
}

func (c *CMAC) MAC(m []byte) [aes.BlockSize]byte {
	fips140.RecordApproved()
	_ = c.b // Hoist the nil check out of the loop.
	var x [aes.BlockSize]byte
	if len(m) == 0 {
		// Special-cased as a single empty partial final block.
		x = c.k2
		x[len(m)] ^= 0b10000000
		c.b.Encrypt(x[:], x[:])
		return x
	}
	for len(m) >= aes.BlockSize {
		subtle.XORBytes(x[:], m[:aes.BlockSize], x[:])
		if len(m) == aes.BlockSize {
			// Final complete block.
			subtle.XORBytes(x[:], c.k1[:], x[:])
		}
		c.b.Encrypt(x[:], x[:])
		m = m[aes.BlockSize:]
	}
	if len(m) > 0 {
		// Final incomplete block.
		subtle.XORBytes(x[:], m, x[:])
		subtle.XORBytes(x[:], c.k2[:], x[:])
		x[len(m)] ^= 0b10000000
		c.b.Encrypt(x[:], x[:])
	}
	return x
}

// shiftLeft sets x to x << 1, and returns MSB₁(x).
func shiftLeft(x *[aes.BlockSize]byte) byte {
	var msb byte
	for i := len(x) - 1; i >= 0; i-- {
		msb, x[i] = x[i]>>7, x[i]<<1|msb
	}
	return msb
}
