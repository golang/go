// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/internal/cryptotest"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"testing"
)

// cfbTests contains the test vectors from
// https://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf, section
// F.3.13.
var cfbTests = []struct {
	key, iv, plaintext, ciphertext string
}{
	{
		"2b7e151628aed2a6abf7158809cf4f3c",
		"000102030405060708090a0b0c0d0e0f",
		"6bc1bee22e409f96e93d7e117393172a",
		"3b3fd92eb72dad20333449f8e83cfb4a",
	},
	{
		"2b7e151628aed2a6abf7158809cf4f3c",
		"3B3FD92EB72DAD20333449F8E83CFB4A",
		"ae2d8a571e03ac9c9eb76fac45af8e51",
		"c8a64537a0b3a93fcde3cdad9f1ce58b",
	},
	{
		"2b7e151628aed2a6abf7158809cf4f3c",
		"C8A64537A0B3A93FCDE3CDAD9F1CE58B",
		"30c81c46a35ce411e5fbc1191a0a52ef",
		"26751f67a3cbb140b1808cf187a4f4df",
	},
	{
		"2b7e151628aed2a6abf7158809cf4f3c",
		"26751F67A3CBB140B1808CF187A4F4DF",
		"f69f2445df4f9b17ad2b417be66c3710",
		"c04b05357c5d1c0eeac4c66f9ff7f2e6",
	},
}

func TestCFBVectors(t *testing.T) {
	for i, test := range cfbTests {
		key, err := hex.DecodeString(test.key)
		if err != nil {
			t.Fatal(err)
		}
		iv, err := hex.DecodeString(test.iv)
		if err != nil {
			t.Fatal(err)
		}
		plaintext, err := hex.DecodeString(test.plaintext)
		if err != nil {
			t.Fatal(err)
		}
		expected, err := hex.DecodeString(test.ciphertext)
		if err != nil {
			t.Fatal(err)
		}

		block, err := aes.NewCipher(key)
		if err != nil {
			t.Fatal(err)
		}

		ciphertext := make([]byte, len(plaintext))
		cfb := cipher.NewCFBEncrypter(block, iv)
		cfb.XORKeyStream(ciphertext, plaintext)

		if !bytes.Equal(ciphertext, expected) {
			t.Errorf("#%d: wrong output: got %x, expected %x", i, ciphertext, expected)
		}

		cfbdec := cipher.NewCFBDecrypter(block, iv)
		plaintextCopy := make([]byte, len(ciphertext))
		cfbdec.XORKeyStream(plaintextCopy, ciphertext)

		if !bytes.Equal(plaintextCopy, plaintext) {
			t.Errorf("#%d: wrong plaintext: got %x, expected %x", i, plaintextCopy, plaintext)
		}
	}
}

func TestCFBInverse(t *testing.T) {
	block, err := aes.NewCipher(commonKey128)
	if err != nil {
		t.Error(err)
		return
	}

	plaintext := []byte("this is the plaintext. this is the plaintext.")
	iv := make([]byte, block.BlockSize())
	rand.Reader.Read(iv)
	cfb := cipher.NewCFBEncrypter(block, iv)
	ciphertext := make([]byte, len(plaintext))
	copy(ciphertext, plaintext)
	cfb.XORKeyStream(ciphertext, ciphertext)

	cfbdec := cipher.NewCFBDecrypter(block, iv)
	plaintextCopy := make([]byte, len(plaintext))
	copy(plaintextCopy, ciphertext)
	cfbdec.XORKeyStream(plaintextCopy, plaintextCopy)

	if !bytes.Equal(plaintextCopy, plaintext) {
		t.Errorf("got: %x, want: %x", plaintextCopy, plaintext)
	}
}

func TestCFBStream(t *testing.T) {

	for _, keylen := range []int{128, 192, 256} {

		t.Run(fmt.Sprintf("AES-%d", keylen), func(t *testing.T) {
			rng := newRandReader(t)

			key := make([]byte, keylen/8)
			rng.Read(key)

			block, err := aes.NewCipher(key)
			if err != nil {
				panic(err)
			}

			t.Run("Encrypter", func(t *testing.T) {
				cryptotest.TestStreamFromBlock(t, block, cipher.NewCFBEncrypter)
			})
			t.Run("Decrypter", func(t *testing.T) {
				cryptotest.TestStreamFromBlock(t, block, cipher.NewCFBDecrypter)
			})
		})
	}

	t.Run("DES", func(t *testing.T) {
		rng := newRandReader(t)

		key := make([]byte, 8)
		rng.Read(key)

		block, err := des.NewCipher(key)
		if err != nil {
			panic(err)
		}

		t.Run("Encrypter", func(t *testing.T) {
			cryptotest.TestStreamFromBlock(t, block, cipher.NewCFBEncrypter)
		})
		t.Run("Decrypter", func(t *testing.T) {
			cryptotest.TestStreamFromBlock(t, block, cipher.NewCFBDecrypter)
		})
	})
}
