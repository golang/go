// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package aes

import (
	"crypto/cipher"
	"crypto/subtle"
	"errors"
)

// The following functions are defined in gcm_amd64.s.
func hasGCMAsm() bool

//go:noescape
func aesEncBlock(dst, src *[16]byte, ks []uint32)

//go:noescape
func gcmAesInit(productTable *[256]byte, ks []uint32)

//go:noescape
func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)

//go:noescape
func gcmAesEnc(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)

//go:noescape
func gcmAesDec(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)

//go:noescape
func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte, pLen, dLen uint64)

const (
	gcmBlockSize         = 16
	gcmTagSize           = 16
	gcmStandardNonceSize = 12
)

var errOpen = errors.New("cipher: message authentication failed")

// aesCipherGCM implements crypto/cipher.gcmAble so that crypto/cipher.NewGCM
// will use the optimised implementation in this file when possible. Instances
// of this type only exist when hasGCMAsm returns true.
type aesCipherGCM struct {
	aesCipherAsm
}

// Assert that aesCipherGCM implements the gcmAble interface.
var _ gcmAble = (*aesCipherGCM)(nil)

// NewGCM returns the AES cipher wrapped in Galois Counter Mode. This is only
// called by crypto/cipher.NewGCM via the gcmAble interface.
func (c *aesCipherGCM) NewGCM(nonceSize int) (cipher.AEAD, error) {
	g := &gcmAsm{ks: c.enc, nonceSize: nonceSize}
	gcmAesInit(&g.productTable, g.ks)
	return g, nil
}

type gcmAsm struct {
	// ks is the key schedule, the length of which depends on the size of
	// the AES key.
	ks []uint32
	// productTable contains pre-computed multiples of the binary-field
	// element used in GHASH.
	productTable [256]byte
	// nonceSize contains the expected size of the nonce, in bytes.
	nonceSize int
}

func (g *gcmAsm) NonceSize() int {
	return g.nonceSize
}

func (*gcmAsm) Overhead() int {
	return gcmTagSize
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

// Seal encrypts and authenticates plaintext. See the cipher.AEAD interface for
// details.
func (g *gcmAsm) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}

	var counter, tagMask [gcmBlockSize]byte

	if len(nonce) == gcmStandardNonceSize {
		// Init counter to nonce||1
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		// Otherwise counter = GHASH(nonce)
		gcmAesData(&g.productTable, nonce, &counter)
		gcmAesFinish(&g.productTable, &tagMask, &counter, uint64(len(nonce)), uint64(0))
	}

	aesEncBlock(&tagMask, &counter, g.ks)

	var tagOut [gcmTagSize]byte
	gcmAesData(&g.productTable, data, &tagOut)

	ret, out := sliceForAppend(dst, len(plaintext)+gcmTagSize)
	if len(plaintext) > 0 {
		gcmAesEnc(&g.productTable, out, plaintext, &counter, &tagOut, g.ks)
	}
	gcmAesFinish(&g.productTable, &tagMask, &tagOut, uint64(len(plaintext)), uint64(len(data)))
	copy(out[len(plaintext):], tagOut[:])

	return ret
}

// Open authenticates and decrypts ciphertext. See the cipher.AEAD interface
// for details.
func (g *gcmAsm) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}

	if len(ciphertext) < gcmTagSize {
		return nil, errOpen
	}
	tag := ciphertext[len(ciphertext)-gcmTagSize:]
	ciphertext = ciphertext[:len(ciphertext)-gcmTagSize]

	// See GCM spec, section 7.1.
	var counter, tagMask [gcmBlockSize]byte

	if len(nonce) == gcmStandardNonceSize {
		// Init counter to nonce||1
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		// Otherwise counter = GHASH(nonce)
		gcmAesData(&g.productTable, nonce, &counter)
		gcmAesFinish(&g.productTable, &tagMask, &counter, uint64(len(nonce)), uint64(0))
	}

	aesEncBlock(&tagMask, &counter, g.ks)

	var expectedTag [gcmTagSize]byte
	gcmAesData(&g.productTable, data, &expectedTag)

	ret, out := sliceForAppend(dst, len(ciphertext))
	if len(ciphertext) > 0 {
		gcmAesDec(&g.productTable, out, ciphertext, &counter, &expectedTag, g.ks)
	}
	gcmAesFinish(&g.productTable, &tagMask, &expectedTag, uint64(len(ciphertext)), uint64(len(data)))

	if subtle.ConstantTimeCompare(expectedTag[:], tag) != 1 {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	return ret, nil
}
