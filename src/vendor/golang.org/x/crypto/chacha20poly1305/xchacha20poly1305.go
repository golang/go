// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha20poly1305

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"

	"golang.org/x/crypto/internal/chacha20"
)

type xchacha20poly1305 struct {
	key [8]uint32
}

// NewX returns a XChaCha20-Poly1305 AEAD that uses the given 256-bit key.
//
// XChaCha20-Poly1305 is a ChaCha20-Poly1305 variant that takes a longer nonce,
// suitable to be generated randomly without risk of collisions. It should be
// preferred when nonce uniqueness cannot be trivially ensured, or whenever
// nonces are randomly generated.
func NewX(key []byte) (cipher.AEAD, error) {
	if len(key) != KeySize {
		return nil, errors.New("chacha20poly1305: bad key length")
	}
	ret := new(xchacha20poly1305)
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

func (*xchacha20poly1305) NonceSize() int {
	return NonceSizeX
}

func (*xchacha20poly1305) Overhead() int {
	return 16
}

func (x *xchacha20poly1305) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != NonceSizeX {
		panic("chacha20poly1305: bad nonce length passed to Seal")
	}

	// XChaCha20-Poly1305 technically supports a 64-bit counter, so there is no
	// size limit. However, since we reuse the ChaCha20-Poly1305 implementation,
	// the second half of the counter is not available. This is unlikely to be
	// an issue because the cipher.AEAD API requires the entire message to be in
	// memory, and the counter overflows at 256 GB.
	if uint64(len(plaintext)) > (1<<38)-64 {
		panic("chacha20poly1305: plaintext too large")
	}

	hNonce := [4]uint32{
		binary.LittleEndian.Uint32(nonce[0:4]),
		binary.LittleEndian.Uint32(nonce[4:8]),
		binary.LittleEndian.Uint32(nonce[8:12]),
		binary.LittleEndian.Uint32(nonce[12:16]),
	}
	c := &chacha20poly1305{
		key: chacha20.HChaCha20(&x.key, &hNonce),
	}
	// The first 4 bytes of the final nonce are unused counter space.
	cNonce := make([]byte, NonceSize)
	copy(cNonce[4:12], nonce[16:24])

	return c.seal(dst, cNonce[:], plaintext, additionalData)
}

func (x *xchacha20poly1305) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != NonceSizeX {
		panic("chacha20poly1305: bad nonce length passed to Open")
	}
	if len(ciphertext) < 16 {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > (1<<38)-48 {
		panic("chacha20poly1305: ciphertext too large")
	}

	hNonce := [4]uint32{
		binary.LittleEndian.Uint32(nonce[0:4]),
		binary.LittleEndian.Uint32(nonce[4:8]),
		binary.LittleEndian.Uint32(nonce[8:12]),
		binary.LittleEndian.Uint32(nonce[12:16]),
	}
	c := &chacha20poly1305{
		key: chacha20.HChaCha20(&x.key, &hNonce),
	}
	// The first 4 bytes of the final nonce are unused counter space.
	cNonce := make([]byte, NonceSize)
	copy(cNonce[4:12], nonce[16:24])

	return c.open(dst, cNonce[:], ciphertext, additionalData)
}
