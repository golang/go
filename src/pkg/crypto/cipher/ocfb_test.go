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

func testOCFB(t *testing.T, resync OCFBResyncOption) {
	block, err := aes.NewCipher(commonKey128)
	if err != nil {
		t.Error(err)
		return
	}

	plaintext := []byte("this is the plaintext, which is long enough to span several blocks.")
	randData := make([]byte, block.BlockSize())
	rand.Reader.Read(randData)
	ocfb, prefix := NewOCFBEncrypter(block, randData, resync)
	ciphertext := make([]byte, len(plaintext))
	ocfb.XORKeyStream(ciphertext, plaintext)

	ocfbdec := NewOCFBDecrypter(block, prefix, resync)
	if ocfbdec == nil {
		t.Errorf("NewOCFBDecrypter failed (resync: %t)", resync)
		return
	}
	plaintextCopy := make([]byte, len(plaintext))
	ocfbdec.XORKeyStream(plaintextCopy, ciphertext)

	if !bytes.Equal(plaintextCopy, plaintext) {
		t.Errorf("got: %x, want: %x (resync: %t)", plaintextCopy, plaintext, resync)
	}
}

func TestOCFB(t *testing.T) {
	testOCFB(t, OCFBNoResync)
	testOCFB(t, OCFBResync)
}
