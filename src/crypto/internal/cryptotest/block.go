// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"crypto/cipher"
	"testing"
)

type MakeBlock func(key []byte) (cipher.Block, error)

// TestBlock performs a set of tests on cipher.Block implementations, checking
// the documented requirements of BlockSize, Encrypt, and Decrypt.
func TestBlock(t *testing.T, keySize int, mb MakeBlock) {
	// Generate random key
	key := make([]byte, keySize)
	newRandReader(t).Read(key)
	t.Logf("Cipher key: 0x%x", key)

	block, err := mb(key)
	if err != nil {
		t.Fatal(err)
	}

	blockSize := block.BlockSize()

	t.Run("Encryption", func(t *testing.T) {
		testCipher(t, block.Encrypt, blockSize)
	})

	t.Run("Decryption", func(t *testing.T) {
		testCipher(t, block.Decrypt, blockSize)
	})

	// Checks baseline Encrypt/Decrypt functionality.  More thorough
	// implementation-specific characterization/golden tests should be done
	// for each block cipher implementation.
	t.Run("Roundtrip", func(t *testing.T) {
		rng := newRandReader(t)

		// Check Decrypt inverts Encrypt
		before, ciphertext, after := make([]byte, blockSize), make([]byte, blockSize), make([]byte, blockSize)

		rng.Read(before)

		block.Encrypt(ciphertext, before)
		block.Decrypt(after, ciphertext)

		if !bytes.Equal(after, before) {
			t.Errorf("plaintext is different after an encrypt/decrypt cycle; got %x, want %x", after, before)
		}

		// Check Encrypt inverts Decrypt (assumes block ciphers are deterministic)
		before, plaintext, after := make([]byte, blockSize), make([]byte, blockSize), make([]byte, blockSize)

		rng.Read(before)

		block.Decrypt(plaintext, before)
		block.Encrypt(after, plaintext)

		if !bytes.Equal(after, before) {
			t.Errorf("ciphertext is different after a decrypt/encrypt cycle; got %x, want %x", after, before)
		}
	})

}

func testCipher(t *testing.T, cipher func(dst, src []byte), blockSize int) {
	t.Run("AlterInput", func(t *testing.T) {
		rng := newRandReader(t)

		// Make long src that shouldn't be modified at all, within block
		// size scope or beyond it
		src, before := make([]byte, blockSize*2), make([]byte, blockSize*2)
		rng.Read(src)
		copy(before, src)

		dst := make([]byte, blockSize)

		cipher(dst, src)
		if !bytes.Equal(src, before) {
			t.Errorf("block cipher modified src; got %x, want %x", src, before)
		}
	})

	t.Run("Aliasing", func(t *testing.T) {
		rng := newRandReader(t)

		buff, expectedOutput := make([]byte, blockSize), make([]byte, blockSize)

		// Record what output is when src and dst are different
		rng.Read(buff)
		cipher(expectedOutput, buff)

		// Check that the same output is generated when src=dst alias to the same
		// memory
		cipher(buff, buff)
		if !bytes.Equal(buff, expectedOutput) {
			t.Errorf("block cipher produced different output when dst = src; got %x, want %x", buff, expectedOutput)
		}
	})

	t.Run("OutOfBoundsWrite", func(t *testing.T) {
		rng := newRandReader(t)

		src := make([]byte, blockSize)
		rng.Read(src)

		// Make a buffer with dst in the middle and data on either end
		buff := make([]byte, blockSize*3)
		endOfPrefix, startOfSuffix := blockSize, blockSize*2
		rng.Read(buff[:endOfPrefix])
		rng.Read(buff[startOfSuffix:])
		dst := buff[endOfPrefix:startOfSuffix]

		// Record the prefix and suffix data to make sure they aren't written to
		initPrefix, initSuffix := make([]byte, blockSize), make([]byte, blockSize)
		copy(initPrefix, buff[:endOfPrefix])
		copy(initSuffix, buff[startOfSuffix:])

		// Write to dst (the middle of the buffer) and make sure it doesn't write
		// beyond the dst slice
		cipher(dst, src)
		if !bytes.Equal(buff[startOfSuffix:], initSuffix) {
			t.Errorf("block cipher did out of bounds write after end of dst slice; got %x, want %x", buff[startOfSuffix:], initSuffix)
		}
		if !bytes.Equal(buff[:endOfPrefix], initPrefix) {
			t.Errorf("block cipher did out of bounds write before beginning of dst slice; got %x, want %x", buff[:endOfPrefix], initPrefix)
		}

		// Check that dst isn't written to beyond BlockSize even if there is room
		// in the slice
		dst = buff[endOfPrefix:] // Extend dst to include suffix
		cipher(dst, src)
		if !bytes.Equal(buff[startOfSuffix:], initSuffix) {
			t.Errorf("block cipher modified dst past BlockSize bytes; got %x, want %x", buff[startOfSuffix:], initSuffix)
		}
	})

	// Check that output of cipher isn't affected by adjacent data beyond input
	// slice scope
	// For encryption, this assumes block ciphers encrypt deterministically
	t.Run("OutOfBoundsRead", func(t *testing.T) {
		rng := newRandReader(t)

		src := make([]byte, blockSize)
		rng.Read(src)
		expectedDst := make([]byte, blockSize)
		cipher(expectedDst, src)

		// Make a buffer with src in the middle and data on either end
		buff := make([]byte, blockSize*3)
		endOfPrefix, startOfSuffix := blockSize, blockSize*2

		copy(buff[endOfPrefix:startOfSuffix], src)
		rng.Read(buff[:endOfPrefix])
		rng.Read(buff[startOfSuffix:])

		testDst := make([]byte, blockSize)
		cipher(testDst, buff[endOfPrefix:startOfSuffix])
		if !bytes.Equal(testDst, expectedDst) {
			t.Errorf("block cipher affected by data outside of src slice bounds; got %x, want %x", testDst, expectedDst)
		}

		// Check that src isn't read from beyond BlockSize even if the slice is
		// longer and contains data in the suffix
		cipher(testDst, buff[endOfPrefix:]) // Input long src
		if !bytes.Equal(testDst, expectedDst) {
			t.Errorf("block cipher affected by src data beyond BlockSize bytes; got %x, want %x", buff[startOfSuffix:], expectedDst)
		}
	})

	t.Run("NonZeroDst", func(t *testing.T) {
		rng := newRandReader(t)

		// Record what the cipher writes into a destination of zeroes
		src := make([]byte, blockSize)
		rng.Read(src)
		expectedDst := make([]byte, blockSize)

		cipher(expectedDst, src)

		// Make nonzero dst
		dst := make([]byte, blockSize*2)
		rng.Read(dst)

		// Remember the random suffix which shouldn't be written to
		expectedDst = append(expectedDst, dst[blockSize:]...)

		cipher(dst, src)
		if !bytes.Equal(dst, expectedDst) {
			t.Errorf("block cipher behavior differs when given non-zero dst; got %x, want %x", dst, expectedDst)
		}
	})

	t.Run("BufferOverlap", func(t *testing.T) {
		rng := newRandReader(t)

		buff := make([]byte, blockSize*2)
		rng.Read((buff))

		// Make src and dst slices point to same array with inexact overlap
		src := buff[:blockSize]
		dst := buff[1 : blockSize+1]
		mustPanic(t, "invalid buffer overlap", func() { cipher(dst, src) })

		// Only overlap on one byte
		src = buff[:blockSize]
		dst = buff[blockSize-1 : 2*blockSize-1]
		mustPanic(t, "invalid buffer overlap", func() { cipher(dst, src) })

		// src comes after dst with one byte overlap
		src = buff[blockSize-1 : 2*blockSize-1]
		dst = buff[:blockSize]
		mustPanic(t, "invalid buffer overlap", func() { cipher(dst, src) })
	})

	// Test short input/output.
	// Assembly used to not notice.
	// See issue 7928.
	t.Run("ShortBlock", func(t *testing.T) {
		// Returns slice of n bytes of an n+1 length array.  Lets us test that a
		// slice is still considered too short even if the underlying array it
		// points to is large enough
		byteSlice := func(n int) []byte { return make([]byte, n+1)[0:n] }

		// Off by one byte
		mustPanic(t, "input not full block", func() { cipher(byteSlice(blockSize), byteSlice(blockSize-1)) })
		mustPanic(t, "output not full block", func() { cipher(byteSlice(blockSize-1), byteSlice(blockSize)) })

		// Small slices
		mustPanic(t, "input not full block", func() { cipher(byteSlice(1), byteSlice(1)) })
		mustPanic(t, "input not full block", func() { cipher(byteSlice(100), byteSlice(1)) })
		mustPanic(t, "output not full block", func() { cipher(byteSlice(1), byteSlice(100)) })
	})
}

func mustPanic(t *testing.T, msg string, f func()) {
	t.Helper()

	defer func() {
		t.Helper()

		err := recover()

		if err == nil {
			t.Errorf("function did not panic for %q", msg)
		}
	}()
	f()
}
