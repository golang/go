// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/internal/fips/alias"
	"strconv"
)

// BlockSize is the AES block size in bytes.
const BlockSize = 16

// A Block is an instance of AES using a particular key.
// It is safe for concurrent use.
type Block struct {
	block
}

// blockExpanded is the block type used for all architectures except s390x,
// which feeds the raw key directly to its instructions.
type blockExpanded struct {
	rounds int
	// Round keys, where only the first (rounds + 1) × (128 ÷ 32) words are used.
	enc [60]uint32
	dec [60]uint32
}

const (
	// AES-128 has 128-bit keys, 10 rounds, and uses 11 128-bit round keys
	// (11×128÷32 = 44 32-bit words).

	// AES-192 has 192-bit keys, 12 rounds, and uses 13 128-bit round keys
	// (13×128÷32 = 52 32-bit words).

	// AES-256 has 256-bit keys, 14 rounds, and uses 15 128-bit round keys
	// (15×128÷32 = 60 32-bit words).

	aes128KeySize = 16
	aes192KeySize = 24
	aes256KeySize = 32

	aes128Rounds = 10
	aes192Rounds = 12
	aes256Rounds = 14
)

// roundKeysSize returns the number of uint32 of c.end or c.dec that are used.
func (b *blockExpanded) roundKeysSize() int {
	return (b.rounds + 1) * (128 / 32)
}

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

// New creates and returns a new [cipher.Block] implementation.
// The key argument should be the AES key, either 16, 24, or 32 bytes to select
// AES-128, AES-192, or AES-256.
func New(key []byte) (*Block, error) {
	// This call is outline to let the allocation happen on the parent stack.
	return newOutlined(&Block{}, key)
}

// newOutlined is marked go:noinline to avoid it inlining into New, and making New
// too complex to inline itself.
//
//go:noinline
func newOutlined(b *Block, key []byte) (*Block, error) {
	switch len(key) {
	case aes128KeySize, aes192KeySize, aes256KeySize:
	default:
		return nil, KeySizeError(len(key))
	}
	return newBlock(b, key), nil
}

func newBlockExpanded(c *blockExpanded, key []byte) {
	switch len(key) {
	case aes128KeySize:
		c.rounds = aes128Rounds
	case aes192KeySize:
		c.rounds = aes192Rounds
	case aes256KeySize:
		c.rounds = aes256Rounds
	}
	expandKeyGeneric(c, key)
}

func (c *Block) BlockSize() int { return BlockSize }

func (c *Block) Encrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	if alias.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	encryptBlock(c, dst, src)
}

func (c *Block) Decrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	if alias.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	decryptBlock(c, dst, src)
}
