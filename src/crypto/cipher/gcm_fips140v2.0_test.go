// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package cipher_test

import (
	"crypto/cipher"
	"crypto/internal/fips140"
	fipsaes "crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"encoding/binary"
	"math"
	"testing"
)

func TestGCMNoncesFIPSV2(t *testing.T) {
	tryNonce := func(aead cipher.AEAD, nonce []byte) bool {
		fips140.ResetServiceIndicator()
		aead.Seal(nil, nonce, []byte("x"), nil)
		return fips140.ServiceIndicator()
	}
	expectOK := func(t *testing.T, aead cipher.AEAD, nonce []byte) {
		t.Helper()
		if !tryNonce(aead, nonce) {
			t.Errorf("expected service indicator true for %x", nonce)
		}
	}
	expectPanic := func(t *testing.T, aead cipher.AEAD, nonce []byte) {
		t.Helper()
		defer func() {
			t.Helper()
			if recover() == nil {
				t.Errorf("expected panic for %x", nonce)
			}
		}()
		tryNonce(aead, nonce)
	}

	t.Run("NewGCMWithXORCounterNonce", func(t *testing.T) {
		newGCM := func() *gcm.GCMWithXORCounterNonce {
			key := make([]byte, 16)
			block, _ := fipsaes.New(key)
			aead, _ := gcm.NewGCMWithXORCounterNonce(block)
			return aead
		}
		nonce := func(mask []byte, counter uint64) []byte {
			nonce := make([]byte, 12)
			copy(nonce, mask)
			n := binary.BigEndian.AppendUint64(nil, counter)
			for i, b := range n {
				nonce[4+i] ^= b
			}
			return nonce
		}

		for _, mask := range [][]byte{
			decodeHex(t, "ffffffffffffffffffffffff"),
			decodeHex(t, "aabbccddeeff001122334455"),
			decodeHex(t, "000000000000000000000000"),
		} {
			g := newGCM()
			// Mask is derived from first invocation with zero nonce.
			expectOK(t, g, nonce(mask, 0))
			expectOK(t, g, nonce(mask, 1))
			expectOK(t, g, nonce(mask, 100))
			expectPanic(t, g, nonce(mask, 100))
			expectPanic(t, g, nonce(mask, 99))
			expectOK(t, g, nonce(mask, math.MaxUint64-2))
			expectOK(t, g, nonce(mask, math.MaxUint64-1))
			expectPanic(t, g, nonce(mask, math.MaxUint64))
			expectPanic(t, g, nonce(mask, 0))

			g = newGCM()
			g.SetNoncePrefixAndMask(mask)
			expectOK(t, g, nonce(mask, 0xFFFFFFFF))
			expectOK(t, g, nonce(mask, math.MaxUint64-2))
			expectOK(t, g, nonce(mask, math.MaxUint64-1))
			expectPanic(t, g, nonce(mask, math.MaxUint64))
			expectPanic(t, g, nonce(mask, 0))

			g = newGCM()
			g.SetNoncePrefixAndMask(mask)
			expectOK(t, g, nonce(mask, math.MaxUint64-1))
			expectPanic(t, g, nonce(mask, math.MaxUint64))
			expectPanic(t, g, nonce(mask, 0))
		}
	})
}
