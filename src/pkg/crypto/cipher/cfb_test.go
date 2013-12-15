// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"testing"
)

func TestCFB(t *testing.T) {
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
