// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"crypto/aes"
	"crypto/cipher"
	"strconv"
	"testing"
)

func benchmarkAESGCMSeal(b *testing.B, buf []byte, keySize int) {
	b.ReportAllocs()
	b.SetBytes(int64(len(buf)))

	var key = make([]byte, keySize)
	var nonce [12]byte
	var ad [13]byte
	aes, _ := aes.NewCipher(key[:])
	aesgcm, _ := cipher.NewGCM(aes)
	var out []byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out = aesgcm.Seal(out[:0], nonce[:], buf, ad[:])
	}
}

func benchmarkAESGCMOpen(b *testing.B, buf []byte, keySize int) {
	b.ReportAllocs()
	b.SetBytes(int64(len(buf)))

	var key = make([]byte, keySize)
	var nonce [12]byte
	var ad [13]byte
	aes, _ := aes.NewCipher(key[:])
	aesgcm, _ := cipher.NewGCM(aes)
	var out []byte

	ct := aesgcm.Seal(nil, nonce[:], buf[:], ad[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ = aesgcm.Open(out[:0], nonce[:], ct, ad[:])
	}
}

func BenchmarkAESGCM(b *testing.B) {
	for _, length := range []int{64, 1350, 8 * 1024} {
		b.Run("Open-128-"+strconv.Itoa(length), func { b -> benchmarkAESGCMOpen(b, make([]byte, length), 128/8) })
		b.Run("Seal-128-"+strconv.Itoa(length), func { b -> benchmarkAESGCMSeal(b, make([]byte, length), 128/8) })

		b.Run("Open-256-"+strconv.Itoa(length), func { b -> benchmarkAESGCMOpen(b, make([]byte, length), 256/8) })
		b.Run("Seal-256-"+strconv.Itoa(length), func { b -> benchmarkAESGCMSeal(b, make([]byte, length), 256/8) })
	}
}

func benchmarkAESStream(b *testing.B, mode func(cipher.Block, []byte) cipher.Stream, buf []byte) {
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var iv [16]byte
	aes, _ := aes.NewCipher(key[:])
	stream := mode(aes, iv[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream.XORKeyStream(buf, buf)
	}
}

// If we test exactly 1K blocks, we would generate exact multiples of
// the cipher's block size, and the cipher stream fragments would
// always be wordsize aligned, whereas non-aligned is a more typical
// use-case.
const almost1K = 1024 - 5
const almost8K = 8*1024 - 5

func BenchmarkAESCFBEncrypt1K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewCFBEncrypter, make([]byte, almost1K))
}

func BenchmarkAESCFBDecrypt1K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewCFBDecrypter, make([]byte, almost1K))
}

func BenchmarkAESCFBDecrypt8K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewCFBDecrypter, make([]byte, almost8K))
}

func BenchmarkAESOFB1K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewOFB, make([]byte, almost1K))
}

func BenchmarkAESCTR1K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewCTR, make([]byte, almost1K))
}

func BenchmarkAESCTR8K(b *testing.B) {
	benchmarkAESStream(b, cipher.NewCTR, make([]byte, almost8K))
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
