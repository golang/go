// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test tests some internals of the flate package.
// The tests in package compress/gzip serve as the
// end-to-end test of the decompressor.

package flate

import (
	"bytes"
	"reflect"
	"testing"
)

// The Huffman code lengths used by the fixed-format Huffman blocks.
var fixedHuffmanBits = [...]int{
	// 0-143 length 8
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,

	// 144-255 length 9
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,

	// 256-279 length 7
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7,

	// 280-287 length 8
	8, 8, 8, 8, 8, 8, 8, 8,
}

type InitDecoderTest struct {
	in  []int
	out huffmanDecoder
	ok  bool
}

var initDecoderTests = []*InitDecoderTest{
	// Example from Connell 1973,
	&InitDecoderTest{
		[]int{3, 5, 2, 4, 3, 5, 5, 4, 4, 3, 4, 5},
		huffmanDecoder{
			2, 5,
			[maxCodeLen + 1]int{2: 0, 4, 13, 31},
			[maxCodeLen + 1]int{2: 0, 1, 6, 20},
			// Paper used different code assignment:
			// 2, 9, 4, 0, 10, 8, 3, 7, 1, 5, 11, 6
			// Reordered here so that codes of same length
			// are assigned to increasing numbers.
			[]int{2, 0, 4, 9, 3, 7, 8, 10, 1, 5, 6, 11},
		},
		true,
	},

	// Example from RFC 1951 section 3.2.2
	&InitDecoderTest{
		[]int{2, 1, 3, 3},
		huffmanDecoder{
			1, 3,
			[maxCodeLen + 1]int{1: 0, 2, 7},
			[maxCodeLen + 1]int{1: 0, 1, 4},
			[]int{1, 0, 2, 3},
		},
		true,
	},

	// Second example from RFC 1951 section 3.2.2
	&InitDecoderTest{
		[]int{3, 3, 3, 3, 3, 2, 4, 4},
		huffmanDecoder{
			2, 4,
			[maxCodeLen + 1]int{2: 0, 6, 15},
			[maxCodeLen + 1]int{2: 0, 1, 8},
			[]int{5, 0, 1, 2, 3, 4, 6, 7},
		},
		true,
	},

	// Static Huffman codes (RFC 1951 section 3.2.6)
	&InitDecoderTest{
		fixedHuffmanBits[0:],
		fixedHuffmanDecoder,
		true,
	},

	// Illegal input.
	&InitDecoderTest{
		[]int{},
		huffmanDecoder{},
		false,
	},

	// Illegal input.
	&InitDecoderTest{
		[]int{0, 0, 0, 0, 0, 0, 0},
		huffmanDecoder{},
		false,
	},
}

func TestInitDecoder(t *testing.T) {
	for i, tt := range initDecoderTests {
		var h huffmanDecoder
		if h.init(tt.in) != tt.ok {
			t.Errorf("test %d: init = %v", i, !tt.ok)
			continue
		}
		if !reflect.DeepEqual(&h, &tt.out) {
			t.Errorf("test %d:\nhave %v\nwant %v", i, h, tt.out)
		}
	}
}

func TestUncompressedSource(t *testing.T) {
	decoder := NewReader(bytes.NewBuffer([]byte{0x01, 0x01, 0x00, 0xfe, 0xff, 0x11}))
	output := make([]byte, 1)
	n, error := decoder.Read(output)
	if n != 1 || error != nil {
		t.Fatalf("decoder.Read() = %d, %v, want 1, nil", n, error)
	}
	if output[0] != 0x11 {
		t.Errorf("output[0] = %x, want 0x11", output[0])
	}
}
