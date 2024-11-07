// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm_test

import (
	"bytes"
	"crypto/internal/cryptotest"
	"crypto/internal/fips/aes"
	"crypto/internal/fips/aes/gcm"
	"crypto/internal/fips/sha3"
	"encoding/hex"
	"testing"
)

func TestAllocations(t *testing.T) {
	cryptotest.SkipTestAllocations(t)
	if allocs := testing.AllocsPerRun(10, func() {
		key := make([]byte, 32)
		nonce := make([]byte, 24)
		plaintext := make([]byte, 16)
		aad := make([]byte, 16)
		ciphertext := make([]byte, 0, 16+16)
		ciphertext = xaesSeal(ciphertext, key, nonce, plaintext, aad)
		if _, err := xaesOpen(plaintext[:0], key, nonce, ciphertext, aad); err != nil {
			t.Fatal(err)
		}
	}); allocs > 0 {
		t.Errorf("expected zero allocations, got %0.1f", allocs)
	}
}

// ACVP tests consider fixed data part of the output, not part of the input, and
// all the pre-generated vectors at
// https://github.com/usnistgov/ACVP-Server/blob/3a7333f6/gen-val/json-files/KDF-1.0/expectedResults.json
// have a 32-byte fixed data, while ours is always 14 bytes. Instead, test
// against the XAES-256-GCM vectors, which were tested against OpenSSL's Counter
// KDF. This also ensures the KDF will work for XAES-256-GCM.

func xaesSeal(dst, key, nonce, plaintext, additionalData []byte) []byte {
	c, _ := aes.New(key)
	k := gcm.NewCounterKDF(c).DeriveKey(0x58, [12]byte(nonce))
	n := nonce[12:]
	a, _ := aes.New(k[:])
	g, _ := gcm.New(a, 12, 16)
	return g.Seal(dst, n, plaintext, additionalData)
}

func xaesOpen(dst, key, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	c, _ := aes.New(key)
	k := gcm.NewCounterKDF(c).DeriveKey(0x58, [12]byte(nonce))
	n := nonce[12:]
	a, _ := aes.New(k[:])
	g, _ := gcm.New(a, 12, 16)
	return g.Open(dst, n, ciphertext, additionalData)
}

func TestVectors(t *testing.T) {
	key := bytes.Repeat([]byte{0x01}, 32)
	nonce := []byte("ABCDEFGHIJKLMNOPQRSTUVWX")
	plaintext := []byte("XAES-256-GCM")
	ciphertext := xaesSeal(nil, key, nonce, plaintext, nil)
	expected := "ce546ef63c9cc60765923609b33a9a1974e96e52daf2fcf7075e2271"
	if got := hex.EncodeToString(ciphertext); got != expected {
		t.Errorf("got: %s", got)
	}
	if decrypted, err := xaesOpen(nil, key, nonce, ciphertext, nil); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(plaintext, decrypted) {
		t.Errorf("plaintext and decrypted are not equal")
	}

	key = bytes.Repeat([]byte{0x03}, 32)
	aad := []byte("c2sp.org/XAES-256-GCM")
	ciphertext = xaesSeal(nil, key, nonce, plaintext, aad)
	expected = "986ec1832593df5443a179437fd083bf3fdb41abd740a21f71eb769d"
	if got := hex.EncodeToString(ciphertext); got != expected {
		t.Errorf("got: %s", got)
	}
	if decrypted, err := xaesOpen(nil, key, nonce, ciphertext, aad); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(plaintext, decrypted) {
		t.Errorf("plaintext and decrypted are not equal")
	}
}

func TestAccumulated(t *testing.T) {
	iterations := 10_000
	expected := "e6b9edf2df6cec60c8cbd864e2211b597fb69a529160cd040d56c0c210081939"

	s, d := sha3.NewShake128(), sha3.NewShake128()
	for i := 0; i < iterations; i++ {
		key := make([]byte, 32)
		s.Read(key)
		nonce := make([]byte, 24)
		s.Read(nonce)
		lenByte := make([]byte, 1)
		s.Read(lenByte)
		plaintext := make([]byte, int(lenByte[0]))
		s.Read(plaintext)
		s.Read(lenByte)
		aad := make([]byte, int(lenByte[0]))
		s.Read(aad)

		ciphertext := xaesSeal(nil, key, nonce, plaintext, aad)
		decrypted, err := xaesOpen(nil, key, nonce, ciphertext, aad)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(plaintext, decrypted) {
			t.Errorf("plaintext and decrypted are not equal")
		}

		d.Write(ciphertext)
	}
	if got := hex.EncodeToString(d.Sum(nil)); got != expected {
		t.Errorf("got: %s", got)
	}
}
