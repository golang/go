// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7,amd64,!gccgo,!appengine

package chacha20poly1305

import "encoding/binary"

//go:noescape
func chacha20Poly1305Open(dst []byte, key []uint32, src, ad []byte) bool

//go:noescape
func chacha20Poly1305Seal(dst []byte, key []uint32, src, ad []byte)

// cpuid is implemented in chacha20poly1305_amd64.s.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

// xgetbv with ecx = 0 is implemented in chacha20poly1305_amd64.s.
func xgetbv() (eax, edx uint32)

var (
	useASM  bool
	useAVX2 bool
)

func init() {
	detectCpuFeatures()
}

// detectCpuFeatures is used to detect if cpu instructions
// used by the functions implemented in assembler in
// chacha20poly1305_amd64.s are supported.
func detectCpuFeatures() {
	maxId, _, _, _ := cpuid(0, 0)
	if maxId < 1 {
		return
	}

	_, _, ecx1, _ := cpuid(1, 0)

	haveSSSE3 := isSet(9, ecx1)
	useASM = haveSSSE3

	haveOSXSAVE := isSet(27, ecx1)

	osSupportsAVX := false
	// For XGETBV, OSXSAVE bit is required and sufficient.
	if haveOSXSAVE {
		eax, _ := xgetbv()
		// Check if XMM and YMM registers have OS support.
		osSupportsAVX = isSet(1, eax) && isSet(2, eax)
	}
	haveAVX := isSet(28, ecx1) && osSupportsAVX

	if maxId < 7 {
		return
	}

	_, ebx7, _, _ := cpuid(7, 0)
	haveAVX2 := isSet(5, ebx7) && haveAVX
	haveBMI2 := isSet(8, ebx7)

	useAVX2 = haveAVX2 && haveBMI2
}

// isSet checks if bit at bitpos is set in value.
func isSet(bitpos uint, value uint32) bool {
	return value&(1<<bitpos) != 0
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
	if !useASM {
		return c.sealGeneric(dst, nonce, plaintext, additionalData)
	}

	var state [16]uint32
	setupState(&state, &c.key, nonce)

	ret, out := sliceForAppend(dst, len(plaintext)+16)
	chacha20Poly1305Seal(out[:], state[:], plaintext, additionalData)
	return ret
}

func (c *chacha20poly1305) open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if !useASM {
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
