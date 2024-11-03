// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips/alias"
	"crypto/internal/fips/drbg"
)

// SealWithRandomNonce encrypts plaintext to out, and writes a random nonce to
// nonce. nonce must be 12 bytes, and out must be 16 bytes longer than plaintext.
// out and plaintext may overlap exactly or not at all. additionalData and out
// must not overlap.
//
// This complies with FIPS 140-3 IG C.H Resolution 2.
//
// Note that this is NOT a [cipher.AEAD].Seal method.
func SealWithRandomNonce(g *GCM, nonce, out, plaintext, additionalData []byte) {
	if uint64(len(plaintext)) > uint64((1<<32)-2)*gcmBlockSize {
		panic("crypto/cipher: message too large for GCM")
	}
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCMWithRandomNonce")
	}
	if len(out) != len(plaintext)+gcmTagSize {
		panic("crypto/cipher: incorrect output length given to GCMWithRandomNonce")
	}
	if alias.InexactOverlap(out, plaintext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}
	drbg.Read(nonce)
	seal(out, g, nonce, plaintext, additionalData)
}
