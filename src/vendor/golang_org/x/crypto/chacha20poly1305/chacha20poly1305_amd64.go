// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,go1.7

package chacha20poly1305

import "encoding/binary"

//go:noescape
func chacha20Poly1305Open(dst []byte, key []uint32, src, ad []byte) bool

//go:noescape
func chacha20Poly1305Seal(dst []byte, key []uint32, src, ad []byte)

//go:noescape
func haveSSSE3() bool

var canUseASM bool

func init() {
	canUseASM = haveSSSE3()
}

// setupState writes a ChaCha20 input matrix to state. See
// https://tools.ietf.org/html/rfc7539#section-2.3.
func setupState(state *[16]uint32, key *[32]byte, nonce []byte) {
	state[0] = 0x61707865
	state[1] = 0x3320646e
	state[2] = 0x79622d32
	state[3] = 0x6b206574

	state[4] = binary.LittleEndian.Uint32(key[:4])
	state[5] = binary.LittleEndian.Uint32(key[4:8])
	state[6] = binary.LittleEndian.Uint32(key[8:12])
	state[7] = binary.LittleEndian.Uint32(key[12:16])
	state[8] = binary.LittleEndian.Uint32(key[16:20])
	state[9] = binary.LittleEndian.Uint32(key[20:24])
	state[10] = binary.LittleEndian.Uint32(key[24:28])
	state[11] = binary.LittleEndian.Uint32(key[28:32])

	state[12] = 0
	state[13] = binary.LittleEndian.Uint32(nonce[:4])
	state[14] = binary.LittleEndian.Uint32(nonce[4:8])
	state[15] = binary.LittleEndian.Uint32(nonce[8:12])
}

func (c *chacha20poly1305) seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if !canUseASM {
		return c.sealGeneric(dst, nonce, plaintext, additionalData)
	}

	var state [16]uint32
	setupState(&state, &c.key, nonce)

	ret, out := sliceForAppend(dst, len(plaintext)+16)
	chacha20Poly1305Seal(out[:], state[:], plaintext, additionalData)
	return ret
}

func (c *chacha20poly1305) open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if !canUseASM {
		return c.openGeneric(dst, nonce, ciphertext, additionalData)
	}

	var state [16]uint32
	setupState(&state, &c.key, nonce)

	ciphertext = ciphertext[:len(ciphertext)-16]
	ret, out := sliceForAppend(dst, len(ciphertext))
	if !chacha20Poly1305Open(out, state[:], ciphertext, additionalData) {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	return ret, nil
}
