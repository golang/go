// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"testing"
)

// TestCipherReversal tests that each cipher factory produces ciphers that can
// encrypt and decrypt some data successfully.
func TestCipherReversal(t *testing.T) {
	testData := []byte("abcdefghijklmnopqrstuvwxyz012345")
	testKey := []byte("AbCdEfGhIjKlMnOpQrStUvWxYz012345")
	testIv := []byte("sdflkjhsadflkjhasdflkjhsadfklhsa")

	cryptBuffer := make([]byte, 32)

	for name, cipherMode := range cipherModes {
		encrypter, err := cipherMode.createCipher(testKey, testIv)
		if err != nil {
			t.Errorf("failed to create encrypter for %q: %s", name, err)
			continue
		}
		decrypter, err := cipherMode.createCipher(testKey, testIv)
		if err != nil {
			t.Errorf("failed to create decrypter for %q: %s", name, err)
			continue
		}

		copy(cryptBuffer, testData)

		encrypter.XORKeyStream(cryptBuffer, cryptBuffer)
		if name == "none" {
			if !bytes.Equal(cryptBuffer, testData) {
				t.Errorf("encryption made change with 'none' cipher")
				continue
			}
		} else {
			if bytes.Equal(cryptBuffer, testData) {
				t.Errorf("encryption made no change with %q", name)
				continue
			}
		}

		decrypter.XORKeyStream(cryptBuffer, cryptBuffer)
		if !bytes.Equal(cryptBuffer, testData) {
			t.Errorf("decrypted bytes not equal to input with %q", name)
			continue
		}
	}
}

func TestDefaultCiphersExist(t *testing.T) {
	for _, cipherAlgo := range DefaultCipherOrder {
		if _, ok := cipherModes[cipherAlgo]; !ok {
			t.Errorf("default cipher %q is unknown", cipherAlgo)
		}
	}
}
