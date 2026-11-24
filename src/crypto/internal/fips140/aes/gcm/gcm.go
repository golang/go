// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/alias"
	"errors"
)

// GCM represents a Galois Counter Mode with a specific key.
type GCM struct {
	cipher    aes.Block
	nonceSize int
	tagSize   int
	gcmPlatformData
}

func New(cipher *aes.Block, nonceSize, tagSize int) (*GCM, error) {
	// This function is outlined to let the allocation happen on the parent stack.
	return newGCM(&GCM{}, cipher, nonceSize, tagSize)
}

// newGCM is marked go:noinline to avoid it inlining into New, and making New
// too complex to inline itself.
//
//go:noinline
func newGCM(g *GCM, cipher *aes.Block, nonceSize, tagSize int) (*GCM, error) {
	if tagSize < gcmMinimumTagSize || tagSize > gcmBlockSize {
		return nil, errors.New("cipher: incorrect tag size given to GCM")
	}
	if nonceSize <= 0 {
		return nil, errors.New("cipher: the nonce can't have zero length")
	}
	if cipher.BlockSize() != gcmBlockSize {
		return nil, errors.New("cipher: NewGCM requires 128-bit block cipher")
	}
	g.cipher = *cipher
	g.nonceSize = nonceSize
	g.tagSize = tagSize
	initGCM(g)
	return g, nil
}

const (
	gcmBlockSize         = 16
	gcmTagSize           = 16
	gcmMinimumTagSize    = 12 // NIST SP 800-38D recommends tags with 12 or more bytes.
	gcmStandardNonceSize = 12
)

func (g *GCM) NonceSize() int {
	return g.nonceSize
}

func (g *GCM) Overhead() int {
	return g.tagSize
}

func (g *GCM) TagSize() int {
	return g.tagSize
}

func (g *GCM) SealRaw(out, tag, nonce, plaintext, additionalData []byte) error {
	fips140.RecordNonApproved()
	if err := g.checkSealRaw(out, tag, nonce, plaintext, additionalData); err != nil {
		return err
	}
	seal(out, tag, g, nonce, plaintext, additionalData)
	return nil
}

// checkSealRaw checks the arguments of SealRaw.
// If they are invalid, it clears out and tag and returns an error.
func (g *GCM) checkSealRaw(out, tag, nonce, plaintext, additionalData []byte) error {
	if g.nonceSize == 0 || g.tagSize == 0 {
		panic("internal error: uninitialized GCM")
	}
	err := check(out, tag, nonce, plaintext, additionalData, g.tagSize, g.nonceSize)
	if err != nil {
		clear(out)
		clear(tag)
	}
	return err
}

func (g *GCM) OpenRaw(out, nonce, ciphertext, tag, additionalData []byte) error {
	if g.nonceSize == 0 || g.tagSize == 0 {
		panic("internal error: uninitialized GCM")
	}
	err := check(out, tag, nonce, ciphertext, additionalData, g.tagSize, g.nonceSize)
	if err != nil {
		clear(out)
		return err
	}

	fips140.RecordApproved()
	if err := open(out, g, nonce, ciphertext, tag, additionalData); err != nil {
		clear(out)
		return err
	}
	return nil
}

func check(out, tag, nonce, input, additionalData []byte, tagSize, nonceSize int) error {
	// len(out) must equal len(plaintext/ciphertext), len(tag) must equal
	// TagSize(), and len(nonce) must equal NonceSize().
	if len(out) != len(input) {
		return errors.New("incorrect output length for AES-GCM")
	}
	if len(tag) != tagSize {
		return errors.New("incorrect tag length for AES-GCM")
	}
	if len(nonce) != nonceSize {
		return errors.New("incorrect nonce length for AES-GCM")
	}

	if uint64(len(input)) > uint64((1<<32)-2)*gcmBlockSize {
		return errors.New("message too large for AES-GCM")
	}

	// out and plaintext/ciphertext must overlap exactly or not at all; out
	// must not overlap nonce or additionalData; tag must not overlap any other
	// argument.
	if alias.InexactOverlap(out, input) {
		return errors.New("invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, tag) {
		return errors.New("invalid buffer overlap of output and tag")
	}
	if alias.AnyOverlap(out, nonce) {
		return errors.New("invalid buffer overlap of output and nonce")
	}
	if alias.AnyOverlap(out, additionalData) {
		return errors.New("invalid buffer overlap of output and additional data")
	}
	if alias.AnyOverlap(tag, input) {
		return errors.New("invalid buffer overlap of tag and input")
	}
	if alias.AnyOverlap(tag, nonce) {
		return errors.New("invalid buffer overlap of tag and nonce")
	}
	if alias.AnyOverlap(tag, additionalData) {
		return errors.New("invalid buffer overlap of tag and additional data")
	}

	return nil
}

func (g *GCM) Seal(dst, nonce, plaintext, data []byte) []byte {
	fips140.RecordNonApproved()
	return g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCM) sealAfterIndicator(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if g.nonceSize == 0 {
		panic("crypto/cipher: incorrect GCM nonce size")
	}
	if uint64(len(plaintext)) > uint64((1<<32)-2)*gcmBlockSize {
		panic("crypto/cipher: message too large for GCM")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+g.tagSize)
	if alias.InexactOverlap(out, plaintext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, data) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}

	seal(out[:len(plaintext)], out[len(plaintext):], g, nonce, plaintext, data)
	return ret
}

var errOpen = errors.New("cipher: message authentication failed")

func (g *GCM) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	// Sanity check to prevent the authentication from always succeeding if an
	// implementation leaves tagSize uninitialized, for example.
	if g.tagSize < gcmMinimumTagSize {
		panic("crypto/cipher: incorrect GCM tag size")
	}

	if len(ciphertext) < g.tagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > uint64((1<<32)-2)*gcmBlockSize+uint64(g.tagSize) {
		return nil, errOpen
	}

	ret, out := sliceForAppend(dst, len(ciphertext)-g.tagSize)
	if alias.InexactOverlap(out, ciphertext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, data) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	fips140.RecordApproved()
	if err := open(out, g, nonce, ciphertext, tag, data); err != nil {
		// We sometimes decrypt and authenticate concurrently, so we overwrite
		// dst in the event of a tag mismatch. To be consistent across platforms
		// and to avoid releasing unauthenticated plaintext, we clear the buffer
		// in the event of an error.
		clear(out)
		return nil, err
	}
	return ret, nil
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
