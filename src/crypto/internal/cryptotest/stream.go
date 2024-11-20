// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"crypto/cipher"
	"crypto/subtle"
	"fmt"
	"strings"
	"testing"
)

// Each test is executed with each of the buffer lengths in bufLens.
var (
	bufLens = []int{0, 1, 3, 4, 8, 10, 15, 16, 20, 32, 50, 4096, 5000}
	bufCap  = 10000
)

// MakeStream returns a cipher.Stream instance.
//
// Multiple calls to MakeStream must return equivalent instances,
// so for example the key and/or IV must be fixed.
type MakeStream func() cipher.Stream

// TestStream performs a set of tests on cipher.Stream implementations,
// checking the documented requirements of XORKeyStream.
func TestStream(t *testing.T, ms MakeStream) {

	t.Run("XORSemantics", func(t *testing.T) {
		if strings.Contains(t.Name(), "TestCFBStream") {
			// This is ugly, but so is CFB's abuse of cipher.Stream.
			// Don't want to make it easier for anyone else to do that.
			t.Skip("CFB implements cipher.Stream but does not follow XOR semantics")
		}

		// Test that XORKeyStream inverts itself for encryption/decryption.
		t.Run("Roundtrip", func(t *testing.T) {

			for _, length := range bufLens {
				t.Run(fmt.Sprintf("BuffLength=%d", length), func(t *testing.T) {
					rng := newRandReader(t)

					plaintext := make([]byte, length)
					rng.Read(plaintext)

					ciphertext := make([]byte, length)
					decrypted := make([]byte, length)

					ms().XORKeyStream(ciphertext, plaintext) // Encrypt plaintext
					ms().XORKeyStream(decrypted, ciphertext) // Decrypt ciphertext
					if !bytes.Equal(decrypted, plaintext) {
						t.Errorf("plaintext is different after an encrypt/decrypt cycle; got %s, want %s", truncateHex(decrypted), truncateHex(plaintext))
					}
				})
			}
		})

		// Test that XORKeyStream behaves the same as directly XORing
		// plaintext with the stream.
		t.Run("DirectXOR", func(t *testing.T) {

			for _, length := range bufLens {
				t.Run(fmt.Sprintf("BuffLength=%d", length), func(t *testing.T) {
					rng := newRandReader(t)

					plaintext := make([]byte, length)
					rng.Read(plaintext)

					// Encrypting all zeros should reveal the stream itself
					stream, directXOR := make([]byte, length), make([]byte, length)
					ms().XORKeyStream(stream, stream)
					// Encrypt plaintext by directly XORing the stream
					subtle.XORBytes(directXOR, stream, plaintext)

					// Encrypt plaintext with XORKeyStream
					ciphertext := make([]byte, length)
					ms().XORKeyStream(ciphertext, plaintext)
					if !bytes.Equal(ciphertext, directXOR) {
						t.Errorf("xor semantics were not preserved; got %s, want %s", truncateHex(ciphertext), truncateHex(directXOR))
					}
				})
			}
		})
	})

	t.Run("EmptyInput", func(t *testing.T) {
		rng := newRandReader(t)

		src, dst := make([]byte, 100), make([]byte, 100)
		rng.Read(dst)
		before := bytes.Clone(dst)

		ms().XORKeyStream(dst, src[:0])
		if !bytes.Equal(dst, before) {
			t.Errorf("XORKeyStream modified dst on empty input; got %s, want %s", truncateHex(dst), truncateHex(before))
		}
	})

	t.Run("AlterInput", func(t *testing.T) {
		rng := newRandReader(t)
		src, dst, before := make([]byte, bufCap), make([]byte, bufCap), make([]byte, bufCap)
		rng.Read(src)

		for _, length := range bufLens {

			t.Run(fmt.Sprintf("BuffLength=%d", length), func(t *testing.T) {
				copy(before, src)

				ms().XORKeyStream(dst[:length], src[:length])
				if !bytes.Equal(src, before) {
					t.Errorf("XORKeyStream modified src; got %s, want %s", truncateHex(src), truncateHex(before))
				}
			})
		}
	})

	t.Run("Aliasing", func(t *testing.T) {
		rng := newRandReader(t)

		buff, expectedOutput := make([]byte, bufCap), make([]byte, bufCap)

		for _, length := range bufLens {
			// Record what output is when src and dst are different
			rng.Read(buff)
			ms().XORKeyStream(expectedOutput[:length], buff[:length])

			// Check that the same output is generated when src=dst alias to the same
			// memory
			ms().XORKeyStream(buff[:length], buff[:length])
			if !bytes.Equal(buff[:length], expectedOutput[:length]) {
				t.Errorf("block cipher produced different output when dst = src; got %x, want %x", buff[:length], expectedOutput[:length])
			}
		}
	})

	t.Run("OutOfBoundsWrite", func(t *testing.T) { // Issue 21104
		rng := newRandReader(t)

		plaintext := make([]byte, bufCap)
		rng.Read(plaintext)
		ciphertext := make([]byte, bufCap)

		for _, length := range bufLens {
			copy(ciphertext, plaintext) // Reset ciphertext buffer

			t.Run(fmt.Sprintf("BuffLength=%d", length), func(t *testing.T) {
				mustPanic(t, "output smaller than input", func() { ms().XORKeyStream(ciphertext[:length], plaintext) })

				if !bytes.Equal(ciphertext[length:], plaintext[length:]) {
					t.Errorf("XORKeyStream did out of bounds write; got %s, want %s", truncateHex(ciphertext[length:]), truncateHex(plaintext[length:]))
				}
			})
		}
	})

	t.Run("BufferOverlap", func(t *testing.T) {
		rng := newRandReader(t)

		buff := make([]byte, bufCap)
		rng.Read(buff)

		for _, length := range bufLens {
			if length == 0 || length == 1 {
				continue
			}

			t.Run(fmt.Sprintf("BuffLength=%d", length), func(t *testing.T) {
				// Make src and dst slices point to same array with inexact overlap
				src := buff[:length]
				dst := buff[1 : length+1]
				mustPanic(t, "invalid buffer overlap", func() { ms().XORKeyStream(dst, src) })

				// Only overlap on one byte
				src = buff[:length]
				dst = buff[length-1 : 2*length-1]
				mustPanic(t, "invalid buffer overlap", func() { ms().XORKeyStream(dst, src) })

				// src comes after dst with one byte overlap
				src = buff[length-1 : 2*length-1]
				dst = buff[:length]
				mustPanic(t, "invalid buffer overlap", func() { ms().XORKeyStream(dst, src) })
			})
		}
	})

	t.Run("KeepState", func(t *testing.T) {
		rng := newRandReader(t)

		plaintext := make([]byte, bufCap)
		rng.Read(plaintext)
		ciphertext := make([]byte, bufCap)

		// Make one long call to XORKeyStream
		ms().XORKeyStream(ciphertext, plaintext)

		for _, step := range bufLens {
			if step == 0 {
				continue
			}
			stepMsg := fmt.Sprintf("step %d: ", step)

			dst := make([]byte, bufCap)

			// Make a bunch of small calls to (stateful) XORKeyStream
			stream := ms()
			i := 0
			for i+step < len(plaintext) {
				stream.XORKeyStream(dst[i:], plaintext[i:i+step])
				i += step
			}
			stream.XORKeyStream(dst[i:], plaintext[i:])

			if !bytes.Equal(dst, ciphertext) {
				t.Errorf(stepMsg+"successive XORKeyStream calls returned a different result than a single one; got %s, want %s", truncateHex(dst), truncateHex(ciphertext))
			}
		}
	})
}

// TestStreamFromBlock creates a Stream from a cipher.Block used in a
// cipher.BlockMode. It addresses Issue 68377 by checking for a panic when the
// BlockMode uses an IV with incorrect length.
// For a valid IV, it also runs all TestStream tests on the resulting stream.
func TestStreamFromBlock(t *testing.T, block cipher.Block, blockMode func(b cipher.Block, iv []byte) cipher.Stream) {

	t.Run("WrongIVLen", func(t *testing.T) {
		t.Skip("see Issue 68377")

		rng := newRandReader(t)
		iv := make([]byte, block.BlockSize()+1)
		rng.Read(iv)
		mustPanic(t, "IV length must equal block size", func() { blockMode(block, iv) })
	})

	t.Run("BlockModeStream", func(t *testing.T) {
		rng := newRandReader(t)
		iv := make([]byte, block.BlockSize())
		rng.Read(iv)

		TestStream(t, func() cipher.Stream { return blockMode(block, iv) })
	})
}

func truncateHex(b []byte) string {
	numVals := 50

	if len(b) <= numVals {
		return fmt.Sprintf("%x", b)
	}
	return fmt.Sprintf("%x...", b[:numVals])
}
