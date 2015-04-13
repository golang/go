// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test tests some internals of the flate package.
// The tests in package compress/gzip serve as the
// end-to-end test of the decompressor.

package flate

import (
	"bytes"
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
	h := new(huffmanDecoder)
	ok := h.init(bits)
	if ok == true {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue5962(t *testing.T) {
	bits := []int{4, 0, 0, 6, 4, 3, 2, 3, 3, 4, 4, 5, 0, 0, 0, 0,
		5, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11}
	h := new(huffmanDecoder)
	ok := h.init(bits)
	if ok == true {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue6255(t *testing.T) {
	bits1 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11}
	bits2 := []int{11, 13}
	h := new(huffmanDecoder)
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
