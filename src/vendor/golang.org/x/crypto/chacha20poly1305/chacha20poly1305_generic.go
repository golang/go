// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha20poly1305

import (
	"encoding/binary"

	"golang.org/x/crypto/chacha20"
	"golang.org/x/crypto/internal/alias"
	"golang.org/x/crypto/internal/poly1305"
)

func writeWithPadding(p *poly1305.MAC, b []byte) {
	p.Write(b)
	if rem := len(b) % 16; rem != 0 {
		var buf [16]byte
		padLen := 16 - rem
		p.Write(buf[:padLen])
	}
}

func writeUint64(p *poly1305.MAC, n int) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(n))
	p.Write(buf[:])
}

func (c *chacha20poly1305) sealGeneric(dst, nonce, plaintext, additionalData []byte) []byte {
	ret, out := sliceForAppend(dst, len(plaintext)+poly1305.TagSize)
	ciphertext, tag := out[:len(plaintext)], out[len(plaintext):]
	if alias.InexactOverlap(out, plaintext) {
		panic("chacha20poly1305: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("chacha20poly1305: invalid buffer overlap of output and additional data")
	}

	var polyKey [32]byte
	s, _ := chacha20.NewUnauthenticatedCipher(c.key[:], nonce)
	s.XORKeyStream(polyKey[:], polyKey[:])
	s.SetCounter(1) // set the counter to 1, skipping 32 bytes
	s.XORKeyStream(ciphertext, plaintext)

	p := poly1305.New(&polyKey)
	writeWithPadding(p, additionalData)
	writeWithPadding(p, ciphertext)
	writeUint64(p, len(additionalData))
	writeUint64(p, len(plaintext))
	p.Sum(tag[:0])

	return ret
}

func (c *chacha20poly1305) openGeneric(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	tag := ciphertext[len(ciphertext)-16:]
	ciphertext = ciphertext[:len(ciphertext)-16]

	var polyKey [32]byte
	s, _ := chacha20.NewUnauthenticatedCipher(c.key[:], nonce)
	s.XORKeyStream(polyKey[:], polyKey[:])
	s.SetCounter(1) // set the counter to 1, skipping 32 bytes

	p := poly1305.New(&polyKey)
	writeWithPadding(p, additionalData)
	writeWithPadding(p, ciphertext)
	writeUint64(p, len(additionalData))
	writeUint64(p, len(ciphertext))

	ret, out := sliceForAppend(dst, len(ciphertext))
	if alias.InexactOverlap(out, ciphertext) {
		panic("chacha20poly1305: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("chacha20poly1305: invalid buffer overlap of output and additional data")
	}
	if !p.Verify(tag) {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	s.XORKeyStream(out, ciphertext)
	return ret, nil
}
