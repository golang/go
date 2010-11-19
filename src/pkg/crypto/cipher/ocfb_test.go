// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import (
	"bytes"
	"crypto/aes"
	"crypto/rand"
	"testing"
)

func TestOCFB(t *testing.T) {
	block, err := aes.NewCipher(commonKey128)
	if err != nil {
		t.Error(err)
		return
	}

	plaintext := []byte("this is the plaintext")
	randData := make([]byte, block.BlockSize())
	rand.Reader.Read(randData)
	ocfb, prefix := NewOCFBEncrypter(block, randData)
	ciphertext := make([]byte, len(plaintext))
	ocfb.XORKeyStream(ciphertext, plaintext)

	ocfbdec := NewOCFBDecrypter(block, prefix)
	if ocfbdec == nil {
		t.Error("NewOCFBDecrypter failed")
		return
	}
	plaintextCopy := make([]byte, len(plaintext))
	ocfbdec.XORKeyStream(plaintextCopy, ciphertext)

	if !bytes.Equal(plaintextCopy, plaintext) {
		t.Errorf("got: %x, want: %x", plaintextCopy, plaintext)
	}
}
