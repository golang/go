// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"hash"
	"io"
	"math/rand"
	"testing"
	"time"
)

type MakeHash func() hash.Hash

// TestHash performs a set of tests on hash.Hash implementations, checking the
// documented requirements of Write, Sum, Reset, Size, and BlockSize.
func TestHash(t *testing.T, mh MakeHash) {

	// Test that Sum returns an appended digest matching output of Size
	t.Run("SumAppend", func(t *testing.T) {
		h := mh()
		rng := newRandReader(t)

		emptyBuff := []byte("")
		shortBuff := []byte("a")
		longBuff := make([]byte, h.BlockSize()+1)
		rng.Read(longBuff)

		// Set of example strings to append digest to
		prefixes := [][]byte{nil, emptyBuff, shortBuff, longBuff}

		// Go to each string and check digest gets appended to and is correct size.
		for _, prefix := range prefixes {
			h.Reset()

			sum := getSum(t, h, prefix) // Append new digest to prefix

			// Check that Sum didn't alter the prefix
			if !bytes.Equal(sum[0:len(prefix)], prefix) {
				t.Errorf("Sum alters passed buffer instead of appending; got %x, want %x", sum[0:len(prefix)], prefix)
			}

			// Check that the appended sum wasn't affected by the prefix
			if expectedSum := getSum(t, h, nil); !bytes.Equal(sum[len(prefix):], expectedSum) {
				t.Errorf("Sum behavior affected by data in the input buffer; got %x, want %x", sum[len(prefix):], expectedSum)
			}

			// Check size of append
			if got, want := len(sum)-len(prefix), h.Size(); got != want {
				t.Errorf("Sum appends number of bytes != Size; got %v , want %v", got, want)
			}
		}
	})

	// Test that Hash.Write never returns error.
	t.Run("WriteWithoutError", func(t *testing.T) {
		h := mh()
		rng := newRandReader(t)

		emptySlice := []byte("")
		shortSlice := []byte("a")
		longSlice := make([]byte, h.BlockSize()+1)
		rng.Read(longSlice)

		// Set of example strings to append digest to
		slices := [][]byte{emptySlice, shortSlice, longSlice}

		for _, slice := range slices {
			writeToHash(t, h, slice) // Writes and checks Write doesn't error
		}
	})

	t.Run("ResetState", func(t *testing.T) {
		h := mh()
		rng := newRandReader(t)

		emptySum := getSum(t, h, nil)

		// Write to hash and then Reset it and see if Sum is same as emptySum
		writeEx := make([]byte, h.BlockSize())
		rng.Read(writeEx)
		writeToHash(t, h, writeEx)
		h.Reset()
		resetSum := getSum(t, h, nil)

		if !bytes.Equal(emptySum, resetSum) {
			t.Errorf("Reset hash yields different Sum than new hash; got %x, want %x", emptySum, resetSum)
		}
	})

	// Check that Write isn't reading from beyond input slice's bounds
	t.Run("OutOfBoundsRead", func(t *testing.T) {
		h := mh()
		blockSize := h.BlockSize()
		rng := newRandReader(t)

		msg := make([]byte, blockSize)
		rng.Read(msg)
		writeToHash(t, h, msg)
		expectedDigest := getSum(t, h, nil) // Record control digest

		h.Reset()

		// Make a buffer with msg in the middle and data on either end
		buff := make([]byte, blockSize*3)
		endOfPrefix, startOfSuffix := blockSize, blockSize*2

		copy(buff[endOfPrefix:startOfSuffix], msg)
		rng.Read(buff[:endOfPrefix])
		rng.Read(buff[startOfSuffix:])

		writeToHash(t, h, buff[endOfPrefix:startOfSuffix])
		testDigest := getSum(t, h, nil)

		if !bytes.Equal(testDigest, expectedDigest) {
			t.Errorf("Write affected by data outside of input slice bounds; got %x, want %x", testDigest, expectedDigest)
		}
	})

	// Test that multiple calls to Write is stateful
	t.Run("StatefulWrite", func(t *testing.T) {
		h := mh()
		rng := newRandReader(t)

		prefix, suffix := make([]byte, h.BlockSize()), make([]byte, h.BlockSize())
		rng.Read(prefix)
		rng.Read(suffix)

		// Write prefix then suffix sequentially and record resulting hash
		writeToHash(t, h, prefix)
		writeToHash(t, h, suffix)
		serialSum := getSum(t, h, nil)

		h.Reset()

		// Write prefix and suffix at the same time and record resulting hash
		writeToHash(t, h, append(prefix, suffix...))
		compositeSum := getSum(t, h, nil)

		// Check that sequential writing results in the same as writing all at once
		if !bytes.Equal(compositeSum, serialSum) {
			t.Errorf("two successive Write calls resulted in a different Sum than a single one; got %x, want %x", compositeSum, serialSum)
		}
	})
}

// Helper function for writing. Verifies that Write does not error.
func writeToHash(t *testing.T, h hash.Hash, p []byte) {
	t.Helper()

	before := make([]byte, len(p))
	copy(before, p)

	n, err := h.Write(p)
	if err != nil || n != len(p) {
		t.Errorf("Write returned error; got (%v, %v), want (nil, %v)", err, n, len(p))
	}

	if !bytes.Equal(p, before) {
		t.Errorf("Write modified input slice; got %x, want %x", p, before)
	}
}

// Helper function for getting Sum. Checks that Sum doesn't change hash state.
func getSum(t *testing.T, h hash.Hash, buff []byte) []byte {
	t.Helper()

	testBuff := make([]byte, len(buff))
	copy(testBuff, buff)

	sum := h.Sum(buff)
	testSum := h.Sum(testBuff)

	// Check that Sum doesn't change underlying hash state
	if !bytes.Equal(sum, testSum) {
		t.Errorf("successive calls to Sum yield different results; got %x, want %x", sum, testSum)
	}

	return sum
}

func newRandReader(t *testing.T) io.Reader {
	seed := time.Now().UnixNano()
	t.Logf("Deterministic RNG seed: 0x%x", seed)
	return rand.New(rand.NewSource(seed))
}
