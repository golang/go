// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/internal/cryptotest"
	"fmt"
	"testing"
)

type noopBlock int

func (b noopBlock) BlockSize() int        { return int(b) }
func (noopBlock) Encrypt(dst, src []byte) { copy(dst, src) }
func (noopBlock) Decrypt(dst, src []byte) { panic("unreachable") }

func inc(b []byte) {
	for i := len(b) - 1; i >= 0; i++ {
		b[i]++
		if b[i] != 0 {
			break
		}
	}
}

func xor(a, b []byte) {
	for i := range a {
		a[i] ^= b[i]
	}
}

func TestCTR(t *testing.T) {
	for size := 64; size <= 1024; size *= 2 {
		iv := make([]byte, size)
		ctr := cipher.NewCTR(noopBlock(size), iv)
		src := make([]byte, 1024)
		for i := range src {
			src[i] = 0xff
		}
		want := make([]byte, 1024)
		copy(want, src)
		counter := make([]byte, size)
		for i := 1; i < len(want)/size; i++ {
			inc(counter)
			xor(want[i*size:(i+1)*size], counter)
		}
		dst := make([]byte, 1024)
		ctr.XORKeyStream(dst, src)
		if !bytes.Equal(dst, want) {
			t.Errorf("for size %d\nhave %x\nwant %x", size, dst, want)
		}
	}
}

func TestCTRStream(t *testing.T) {
	cryptotest.TestAllImplementations(t, "aes", func(t *testing.T) {
		for _, keylen := range []int{128, 192, 256} {
			t.Run(fmt.Sprintf("AES-%d", keylen), func(t *testing.T) {
				rng := newRandReader(t)

				key := make([]byte, keylen/8)
				rng.Read(key)

				block, err := aes.NewCipher(key)
				if err != nil {
					panic(err)
				}

				cryptotest.TestStreamFromBlock(t, block, cipher.NewCTR)
			})
		}
	})

	t.Run("DES", func(t *testing.T) {
		rng := newRandReader(t)

		key := make([]byte, 8)
		rng.Read(key)

		block, err := des.NewCipher(key)
		if err != nil {
			panic(err)
		}

		cryptotest.TestStreamFromBlock(t, block, cipher.NewCTR)
	})
}
