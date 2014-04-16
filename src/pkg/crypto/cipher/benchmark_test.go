// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"crypto/aes"
	"crypto/cipher"
	"testing"
)

func BenchmarkAESGCMSeal1K(b *testing.B) {
	buf := make([]byte, 1024)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var nonce [12]byte
	aes, _ := aes.NewCipher(key[:])
	aesgcm, _ := cipher.NewGCM(aes)
	var out []byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out = aesgcm.Seal(out[:0], nonce[:], buf, nonce[:])
	}
}

func BenchmarkAESGCMOpen1K(b *testing.B) {
	buf := make([]byte, 1024)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var nonce [12]byte
	aes, _ := aes.NewCipher(key[:])
	aesgcm, _ := cipher.NewGCM(aes)
	var out []byte
	out = aesgcm.Seal(out[:0], nonce[:], buf, nonce[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := aesgcm.Open(buf[:0], nonce[:], out, nonce[:])
		if err != nil {
			b.Errorf("Open: %v", err)
		}
	}
}

// If we test exactly 1K blocks, we would generate exact multiples of
// the cipher's block size, and the cipher stream fragments would
// always be wordsize aligned, whereas non-aligned is a more typical
// use-case.
const almost1K = 1024 - 5

func BenchmarkAESCFBEncrypt1K(b *testing.B) {
	buf := make([]byte, almost1K)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	ctr := cipher.NewCFBEncrypter(aes, iv[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctr.XORKeyStream(buf, buf)
	}
}

func BenchmarkAESCFBDecrypt1K(b *testing.B) {
	buf := make([]byte, almost1K)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	ctr := cipher.NewCFBDecrypter(aes, iv[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctr.XORKeyStream(buf, buf)
	}
}

func BenchmarkAESOFB1K(b *testing.B) {
	buf := make([]byte, almost1K)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	ctr := cipher.NewOFB(aes, iv[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctr.XORKeyStream(buf, buf)
	}
}

func BenchmarkAESCTR1K(b *testing.B) {
	buf := make([]byte, almost1K)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	ctr := cipher.NewCTR(aes, iv[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctr.XORKeyStream(buf, buf)
	}
}

func BenchmarkAESCBCEncrypt1K(b *testing.B) {
	buf := make([]byte, 1024)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	cbc := cipher.NewCBCEncrypter(aes, iv[:])
	for i := 0; i < b.N; i++ {
		cbc.CryptBlocks(buf, buf)
	}
}

func BenchmarkAESCBCDecrypt1K(b *testing.B) {
	buf := make([]byte, 1024)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	cbc := cipher.NewCBCDecrypter(aes, iv[:])
	for i := 0; i < b.N; i++ {
		cbc.CryptBlocks(buf, buf)
	}
}
