// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/internal/cryptotest"
	"fmt"
	"io"
	"math/rand"
	"testing"
	"time"
)

// Test CBC Blockmode against the general cipher.BlockMode interface tester
func TestCBCBlockMode(t *testing.T) {
	for _, keylen := range []int{128, 192, 256} {

		t.Run(fmt.Sprintf("AES-%d", keylen), func(t *testing.T) {
			rng := newRandReader(t)

			key := make([]byte, keylen/8)
			rng.Read(key)

			block, err := aes.NewCipher(key)
			if err != nil {
				panic(err)
			}

			cryptotest.TestBlockMode(t, block, cipher.NewCBCEncrypter, cipher.NewCBCDecrypter)
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

		cryptotest.TestBlockMode(t, block, cipher.NewCBCEncrypter, cipher.NewCBCDecrypter)
	})
}

func newRandReader(t *testing.T) io.Reader {
	seed := time.Now().UnixNano()
	t.Logf("Deterministic RNG seed: 0x%x", seed)
	return rand.New(rand.NewSource(seed))
}
