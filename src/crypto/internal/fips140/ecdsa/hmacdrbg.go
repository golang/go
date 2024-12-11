// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
)

// hmacDRBG is an SP 800-90A Rev. 1 HMAC_DRBG.
//
// It is only intended to be used to generate ECDSA nonces. Since it will be
// instantiated ex-novo for each signature, its Generate function will only be
// invoked once or twice (only for P-256, with probability 2⁻³²).
//
// Per Table 2, it has a reseed interval of 2^48 requests, and a maximum request
// size of 2^19 bits (2^16 bytes, 64 KiB).
type hmacDRBG struct {
	newHMAC func(key []byte) *hmac.HMAC

	hK *hmac.HMAC
	V  []byte

	reseedCounter uint64
}

const (
	reseedInterval = 1 << 48
	maxRequestSize = (1 << 19) / 8
)

// plainPersonalizationString is used by HMAC_DRBG as-is.
type plainPersonalizationString []byte

func (plainPersonalizationString) isPersonalizationString() {}

// Each entry in blockAlignedPersonalizationString is written to the HMAC at a
// block boundary, as specified in draft-irtf-cfrg-det-sigs-with-noise-04,
// Section 4.
type blockAlignedPersonalizationString [][]byte

func (blockAlignedPersonalizationString) isPersonalizationString() {}

type personalizationString interface {
	isPersonalizationString()
}

func newDRBG[H fips140.Hash](hash func() H, entropy, nonce []byte, s personalizationString) *hmacDRBG {
	// HMAC_DRBG_Instantiate_algorithm, per Section 10.1.2.3.
	fips140.RecordApproved()

	d := &hmacDRBG{
		newHMAC: func(key []byte) *hmac.HMAC {
			return hmac.New(hash, key)
		},
	}
	size := hash().Size()

	// K = 0x00 0x00 0x00 ... 0x00
	K := make([]byte, size)

	// V = 0x01 0x01 0x01 ... 0x01
	d.V = bytes.Repeat([]byte{0x01}, size)

	// HMAC_DRBG_Update, per Section 10.1.2.2.
	// K = HMAC (K, V || 0x00 || provided_data)
	h := hmac.New(hash, K)
	h.Write(d.V)
	h.Write([]byte{0x00})
	h.Write(entropy)
	h.Write(nonce)
	switch s := s.(type) {
	case plainPersonalizationString:
		h.Write(s)
	case blockAlignedPersonalizationString:
		l := len(d.V) + 1 + len(entropy) + len(nonce)
		for _, b := range s {
			pad000(h, l)
			h.Write(b)
			l = len(b)
		}
	}
	K = h.Sum(K[:0])
	// V = HMAC (K, V)
	h = hmac.New(hash, K)
	h.Write(d.V)
	d.V = h.Sum(d.V[:0])
	// K = HMAC (K, V || 0x01 || provided_data).
	h.Reset()
	h.Write(d.V)
	h.Write([]byte{0x01})
	h.Write(entropy)
	h.Write(nonce)
	switch s := s.(type) {
	case plainPersonalizationString:
		h.Write(s)
	case blockAlignedPersonalizationString:
		l := len(d.V) + 1 + len(entropy) + len(nonce)
		for _, b := range s {
			pad000(h, l)
			h.Write(b)
			l = len(b)
		}
	}
	K = h.Sum(K[:0])
	// V = HMAC (K, V)
	h = hmac.New(hash, K)
	h.Write(d.V)
	d.V = h.Sum(d.V[:0])

	d.hK = h
	d.reseedCounter = 1
	return d
}

func pad000(h *hmac.HMAC, writtenSoFar int) {
	blockSize := h.BlockSize()
	if rem := writtenSoFar % blockSize; rem != 0 {
		h.Write(make([]byte, blockSize-rem))
	}
}

// Generate produces at most maxRequestSize bytes of random data in out.
func (d *hmacDRBG) Generate(out []byte) {
	// HMAC_DRBG_Generate_algorithm, per Section 10.1.2.5.
	fips140.RecordApproved()

	if len(out) > maxRequestSize {
		panic("ecdsa: internal error: request size exceeds maximum")
	}

	if d.reseedCounter > reseedInterval {
		panic("ecdsa: reseed interval exceeded")
	}

	tlen := 0
	for tlen < len(out) {
		// V = HMAC_K(V)
		// T = T || V
		d.hK.Reset()
		d.hK.Write(d.V)
		d.V = d.hK.Sum(d.V[:0])
		tlen += copy(out[tlen:], d.V)
	}

	// Note that if this function shows up on ECDSA-level profiles, this can be
	// optimized in the common case by deferring the rest to the next Generate
	// call, which will never come in nearly all cases.

	// HMAC_DRBG_Update, per Section 10.1.2.2, without provided_data.
	// K = HMAC (K, V || 0x00)
	d.hK.Reset()
	d.hK.Write(d.V)
	d.hK.Write([]byte{0x00})
	K := d.hK.Sum(nil)
	// V = HMAC (K, V)
	d.hK = d.newHMAC(K)
	d.hK.Write(d.V)
	d.V = d.hK.Sum(d.V[:0])

	d.reseedCounter++
}
