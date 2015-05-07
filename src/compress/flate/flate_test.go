// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test tests some internals of the flate package.
// The tests in package compress/gzip serve as the
// end-to-end test of the decompressor.

package flate

import (
	"bytes"
	"io/ioutil"
	"strings"
	"testing"
)

func TestUncompressedSource(t *testing.T) {
	decoder := NewReader(bytes.NewReader([]byte{0x01, 0x01, 0x00, 0xfe, 0xff, 0x11}))
	output := make([]byte, 1)
	n, error := decoder.Read(output)
	if n != 1 || error != nil {
		t.Fatalf("decoder.Read() = %d, %v, want 1, nil", n, error)
	}
	if output[0] != 0x11 {
		t.Errorf("output[0] = %x, want 0x11", output[0])
	}
}

// The following test should not panic.
func TestIssue5915(t *testing.T) {
	bits := []int{4, 0, 0, 6, 4, 3, 2, 3, 3, 4, 4, 5, 0, 0, 0, 0, 5, 5, 6,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 6, 0, 11, 0, 8, 0, 6, 6, 10, 8}
	var h huffmanDecoder
	if h.init(bits) {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue5962(t *testing.T) {
	bits := []int{4, 0, 0, 6, 4, 3, 2, 3, 3, 4, 4, 5, 0, 0, 0, 0,
		5, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11}
	var h huffmanDecoder
	if h.init(bits) {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue6255(t *testing.T) {
	bits1 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11}
	bits2 := []int{11, 13}
	var h huffmanDecoder
	if !h.init(bits1) {
		t.Fatalf("Given sequence of bits is good and should succeed.")
	}
	if h.init(bits2) {
		t.Fatalf("Given sequence of bits is bad and should not succeed.")
	}
}

func TestInvalidEncoding(t *testing.T) {
	// Initialize Huffman decoder to recognize "0".
	var h huffmanDecoder
	if !h.init([]int{1}) {
		t.Fatal("Failed to initialize Huffman decoder")
	}

	// Initialize decompressor with invalid Huffman coding.
	var f decompressor
	f.r = bytes.NewReader([]byte{0xff})

	_, err := f.huffSym(&h)
	if err == nil {
		t.Fatal("Should have rejected invalid bit sequence")
	}
}

func TestInvalidBits(t *testing.T) {
	oversubscribed := []int{1, 2, 3, 4, 4, 5}
	incomplete := []int{1, 2, 4, 4}
	var h huffmanDecoder
	if h.init(oversubscribed) {
		t.Fatal("Should reject oversubscribed bit-length set")
	}
	if h.init(incomplete) {
		t.Fatal("Should reject incomplete bit-length set")
	}
}

func TestDegenerateHuffmanCoding(t *testing.T) {
	const (
		want = "abcabc"
		// This compressed form has a dynamic Huffman block, even though a
		// sensible encoder would use a literal data block, as the latter is
		// shorter. Still, it is a valid flate compression of "abcabc". It has
		// a degenerate Huffman table with only one coded value: the one
		// non-literal back-ref copy of the first "abc" to the second "abc".
		//
		// To verify that this is decompressible with zlib (the C library),
		// it's easy to use the Python wrapper library:
		// >>> import zlib
		// >>> compressed = "\x0c\xc2...etc...\xff\xff"
		// >>> zlib.decompress(compressed, -15) # negative means no GZIP header.
		// 'abcabc'
		compressed = "\x0c\xc2\x01\x0d\x00\x00\x00\x82\xb0\xac\x4a\xff\x0e\xb0\x7d\x27" +
			"\x06\x00\x00\xff\xff"
	)
	b, err := ioutil.ReadAll(NewReader(strings.NewReader(compressed)))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(b); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
