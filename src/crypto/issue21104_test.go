// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rc4"
	"testing"
)

func TestRC4OutOfBoundsWrite(t *testing.T) {
	// This cipherText is encrypted "0123456789"
	cipherText := []byte{238, 41, 187, 114, 151, 2, 107, 13, 178, 63}
	cipher, err := rc4.NewCipher([]byte{0})
	if err != nil {
		panic(err)
	}
	test(t, "RC4", cipherText, cipher.XORKeyStream)
}
func TestCTROutOfBoundsWrite(t *testing.T) {
	testBlock(t, "CTR", cipher.NewCTR)
}
func TestOFBOutOfBoundsWrite(t *testing.T) {
	testBlock(t, "OFB", cipher.NewOFB)
}
func TestCFBEncryptOutOfBoundsWrite(t *testing.T) {
	testBlock(t, "CFB Encrypt", cipher.NewCFBEncrypter)
}
func TestCFBDecryptOutOfBoundsWrite(t *testing.T) {
	testBlock(t, "CFB Decrypt", cipher.NewCFBDecrypter)
}
func testBlock(t *testing.T, name string, newCipher func(cipher.Block, []byte) cipher.Stream) {
	// This cipherText is encrypted "0123456789"
	cipherText := []byte{86, 216, 121, 231, 219, 191, 26, 12, 176, 117}
	var iv, key [16]byte
	block, err := aes.NewCipher(key[:])
	if err != nil {
		panic(err)
	}
	stream := newCipher(block, iv[:])
	test(t, name, cipherText, stream.XORKeyStream)
}
func test(t *testing.T, name string, cipherText []byte, xor func([]byte, []byte)) {
	want := "abcdefghij"
	plainText := []byte(want)
	shorterLen := len(cipherText) / 2
	defer func() {
		err := recover()
		if err == nil {
			t.Errorf("%v XORKeyStream expected to panic on len(dst) < len(src), but didn't", name)
		}
		const plain = "0123456789"
		if plainText[shorterLen] == plain[shorterLen] {
			t.Errorf("%v XORKeyStream did out of bounds write, want %v, got %v", name, want, string(plainText))
		}
	}()
	xor(plainText[:shorterLen], cipherText)
}
