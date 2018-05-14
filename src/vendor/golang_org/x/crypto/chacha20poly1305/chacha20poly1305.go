// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package chacha20poly1305 implements the ChaCha20-Poly1305 AEAD as specified in RFC 7539.
package chacha20poly1305 // import "golang.org/x/crypto/chacha20poly1305"

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"
)

const (
	// KeySize is the size of the key used by this AEAD, in bytes.
	KeySize = 32
	// NonceSize is the size of the nonce used with this AEAD, in bytes.
	NonceSize = 12
)

type chacha20poly1305 struct {
	key [8]uint32
}

// New returns a ChaCha20-Poly1305 AEAD that uses the given, 256-bit key.
func New(key []byte) (cipher.AEAD, error) {
	if len(key) != KeySize {
		return nil, errors.New("chacha20poly1305: bad key length")
	}
	ret := new(chacha20poly1305)
	ret.key[0] = binary.LittleEndian.Uint32(key[0:4])
	ret.key[1] = binary.LittleEndian.Uint32(key[4:8])
	ret.key[2] = binary.LittleEndian.Uint32(key[8:12])
	ret.key[3] = binary.LittleEndian.Uint32(key[12:16])
	ret.key[4] = binary.LittleEndian.Uint32(key[16:20])
	ret.key[5] = binary.LittleEndian.Uint32(key[20:24])
	ret.key[6] = binary.LittleEndian.Uint32(key[24:28])
	ret.key[7] = binary.LittleEndian.Uint32(key[28:32])
	return ret, nil
}

func (c *chacha20poly1305) NonceSize() int {
	return NonceSize
}

func (c *chacha20poly1305) Overhead() int {
	return 16
}

func (c *chacha20poly1305) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != NonceSize {
		panic("chacha20poly1305: bad nonce length passed to Seal")
	}

	if uint64(len(plaintext)) > (1<<38)-64 {
		panic("chacha20poly1305: plaintext too large")
	}

	return c.seal(dst, nonce, plaintext, additionalData)
}

var errOpen = errors.New("chacha20poly1305: message authentication failed")

func (c *chacha20poly1305) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != NonceSize {
		panic("chacha20poly1305: bad nonce length passed to Open")
	}
	if len(ciphertext) < 16 {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > (1<<38)-48 {
		panic("chacha20poly1305: ciphertext too large")
	}

	return c.open(dst, nonce, ciphertext, additionalData)
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
