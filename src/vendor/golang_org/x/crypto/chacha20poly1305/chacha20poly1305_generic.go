// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha20poly1305

import (
	"encoding/binary"

	"golang_org/x/crypto/chacha20poly1305/internal/chacha20"
	"golang_org/x/crypto/poly1305"
)

func roundTo16(n int) int {
	return 16 * ((n + 15) / 16)
}

func (c *chacha20poly1305) sealGeneric(dst, nonce, plaintext, additionalData []byte) []byte {
	var counter [16]byte
	copy(counter[4:], nonce)

	var polyKey [32]byte
	chacha20.XORKeyStream(polyKey[:], polyKey[:], &counter, &c.key)

	ret, out := sliceForAppend(dst, len(plaintext)+poly1305.TagSize)
	counter[0] = 1
	chacha20.XORKeyStream(out, plaintext, &counter, &c.key)

	polyInput := make([]byte, roundTo16(len(additionalData))+roundTo16(len(plaintext))+8+8)
	copy(polyInput, additionalData)
	copy(polyInput[roundTo16(len(additionalData)):], out[:len(plaintext)])
	binary.LittleEndian.PutUint64(polyInput[len(polyInput)-16:], uint64(len(additionalData)))
	binary.LittleEndian.PutUint64(polyInput[len(polyInput)-8:], uint64(len(plaintext)))

	var tag [poly1305.TagSize]byte
	poly1305.Sum(&tag, polyInput, &polyKey)
	copy(out[len(plaintext):], tag[:])

	return ret
}

func (c *chacha20poly1305) openGeneric(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	var tag [poly1305.TagSize]byte
	copy(tag[:], ciphertext[len(ciphertext)-16:])
	ciphertext = ciphertext[:len(ciphertext)-16]

	var counter [16]byte
	copy(counter[4:], nonce)

	var polyKey [32]byte
	chacha20.XORKeyStream(polyKey[:], polyKey[:], &counter, &c.key)

	polyInput := make([]byte, roundTo16(len(additionalData))+roundTo16(len(ciphertext))+8+8)
	copy(polyInput, additionalData)
	copy(polyInput[roundTo16(len(additionalData)):], ciphertext)
	binary.LittleEndian.PutUint64(polyInput[len(polyInput)-16:], uint64(len(additionalData)))
	binary.LittleEndian.PutUint64(polyInput[len(polyInput)-8:], uint64(len(ciphertext)))

	ret, out := sliceForAppend(dst, len(ciphertext))
	if !poly1305.Verify(&tag, polyInput, &polyKey) {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	counter[0] = 1
	chacha20.XORKeyStream(out, ciphertext, &counter, &c.key)
	return ret, nil
}
