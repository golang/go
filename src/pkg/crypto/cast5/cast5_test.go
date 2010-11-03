// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cast5

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// This test vector is taken from RFC 2144, App B.1.
// Since the other two test vectors are for reduced-round variants, we can't
// use them.
var basicTests = []struct {
	key, plainText, cipherText string
}{
	{
		"0123456712345678234567893456789a",
		"0123456789abcdef",
		"238b4fe5847e44b2",
	},
}

func TestBasic(t *testing.T) {
	for i, test := range basicTests {
		key, _ := hex.DecodeString(test.key)
		plainText, _ := hex.DecodeString(test.plainText)
		expected, _ := hex.DecodeString(test.cipherText)

		c, err := NewCipher(key)
		if err != nil {
			t.Errorf("#%d: failed to create Cipher: %s", i, err)
			continue
		}
		var cipherText [BlockSize]byte
		c.Encrypt(cipherText[:], plainText)
		if !bytes.Equal(cipherText[:], expected) {
			t.Errorf("#%d: got:%x want:%x", i, cipherText, expected)
		}

		var plainTextAgain [BlockSize]byte
		c.Decrypt(plainTextAgain[:], cipherText[:])
		if !bytes.Equal(plainTextAgain[:], plainText) {
			t.Errorf("#%d: got:%x want:%x", i, plainTextAgain, plainText)
		}
	}
}

// TestFull performs the test specified in RFC 2144, App B.2.
// However, due to the length of time taken, it's disabled here and a more
// limited version is included, below.
func TestFull(t *testing.T) {
	// This is too slow for normal testing
	return

	a, b := iterate(1000000)

	const expectedA = "eea9d0a249fd3ba6b3436fb89d6dca92"
	const expectedB = "b2c95eb00c31ad7180ac05b8e83d696e"

	if hex.EncodeToString(a) != expectedA {
		t.Errorf("a: got:%x want:%s", a, expectedA)
	}
	if hex.EncodeToString(b) != expectedB {
		t.Errorf("b: got:%x want:%s", b, expectedB)
	}
}

func iterate(iterations int) ([]byte, []byte) {
	const initValueHex = "0123456712345678234567893456789a"

	initValue, _ := hex.DecodeString(initValueHex)

	var a, b [16]byte
	copy(a[:], initValue)
	copy(b[:], initValue)

	for i := 0; i < iterations; i++ {
		c, _ := NewCipher(b[:])
		c.Encrypt(a[:8], a[:8])
		c.Encrypt(a[8:], a[8:])
		c, _ = NewCipher(a[:])
		c.Encrypt(b[:8], b[:8])
		c.Encrypt(b[8:], b[8:])
	}

	return a[:], b[:]
}

func TestLimited(t *testing.T) {
	a, b := iterate(1000)

	const expectedA = "23f73b14b02a2ad7dfb9f2c35644798d"
	const expectedB = "e5bf37eff14c456a40b21ce369370a9f"

	if hex.EncodeToString(a) != expectedA {
		t.Errorf("a: got:%x want:%s", a, expectedA)
	}
	if hex.EncodeToString(b) != expectedB {
		t.Errorf("b: got:%x want:%s", b, expectedB)
	}
}
