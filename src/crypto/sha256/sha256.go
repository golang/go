// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha256 implements the SHA224 and SHA256 hash algorithms as defined
// in FIPS 180-4.
package sha256

import (
	"crypto"
	"crypto/internal/boring"
	"crypto/internal/fips140/sha256"
	"encoding/hex"
	"hash"
)

func init() {
	crypto.RegisterHash(crypto.SHA224, New224)
	crypto.RegisterHash(crypto.SHA256, New)
}

// The size of a SHA256 checksum in bytes.
const Size = 32

// The size of a SHA224 checksum in bytes.
const Size224 = 28

// The blocksize of SHA256 and SHA224 in bytes.
const BlockSize = 64

// New returns a new [hash.Hash] computing the SHA256 checksum. The Hash
// also implements [encoding.BinaryMarshaler], [encoding.BinaryAppender] and
// [encoding.BinaryUnmarshaler] to marshal and unmarshal the internal
// state of the hash.
func New() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA256()
	}
	return sha256.New()
}

// New224 returns a new [hash.Hash] computing the SHA224 checksum. The Hash
// also implements [encoding.BinaryMarshaler], [encoding.BinaryAppender] and
// [encoding.BinaryUnmarshaler] to marshal and unmarshal the internal
// state of the hash.
func New224() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA224()
	}
	return sha256.New224()
}

// Sum256 returns the SHA256 checksum of the data.
func Sum256(data []byte) [Size]byte {
	if boring.Enabled {
		return boring.SHA256(data)
	}
	h := New()
	h.Write(data)
	var sum [Size]byte
	h.Sum(sum[:0])
	return sum
}

// Sum224 returns the SHA224 checksum of the data.
func Sum224(data []byte) [Size224]byte {
	if boring.Enabled {
		return boring.SHA224(data)
	}
	h := New224()
	h.Write(data)
	var sum [Size224]byte
	h.Sum(sum[:0])
	return sum
}

type SHA256Hash [Size]byte

func (h SHA256Hash) String() string {
	return hex.EncodeToString(h[:])
}

func (h SHA256Hash) Truncate(maxLength int) string {
	full := h.String()
	if len(full) > maxLength {
		return full[:maxLength]
	}
	return full
}

type SHA224Hash [Size224]byte

func (h SHA224Hash) String() string {
	return hex.EncodeToString(h[:])
}

func (h SHA224Hash) Truncate(maxLength int) string {
	full := h.String()
	if len(full) > maxLength {
		return full[:maxLength]
	}
	return full
}
