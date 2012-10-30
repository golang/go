// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"bytes"
	"fmt"
	"testing"
)

// We take the smallest, largest and an arbitrary value for each
// of the UTF-8 sequence lengths.
var testRunes = []rune{
	0x01, 0x0C, 0x7F, // 1-byte sequences
	0x80, 0x100, 0x7FF, // 2-byte sequences
	0x800, 0x999, 0xFFFF, // 3-byte sequences
	0x10000, 0x10101, 0x10FFFF, // 4-byte sequences
	0x200, 0x201, 0x202, 0x210, 0x215, // five entries in one sparse block
}

func makeTestTrie(t *testing.T) trie {
	n := newNode()
	for i, r := range testRunes {
		n.insert(r, uint32(i))
	}
	idx := newTrieBuilder()
	idx.addTrie(n)
	tr, err := idx.generate()
	if err != nil {
		t.Errorf(err.Error())
	}
	return *tr
}

func TestGenerateTrie(t *testing.T) {
	testdata := makeTestTrie(t)
	buf := &bytes.Buffer{}
	testdata.printArrays(buf, "test")
	fmt.Fprintf(buf, "var testTrie = ")
	testdata.printStruct(buf, &trieHandle{19, 0}, "test")
	if output != buf.String() {
		t.Error("output differs")
	}
}

var output = `// testValues: 832 entries, 3328 bytes
// Block 2 is the null block.
var testValues = [832]uint32 {
	// Block 0x0, offset 0x0
	0x000c:0x00000001, 
	// Block 0x1, offset 0x40
	0x007f:0x00000002, 
	// Block 0x2, offset 0x80
	// Block 0x3, offset 0xc0
	0x00c0:0x00000003, 
	// Block 0x4, offset 0x100
	0x0100:0x00000004, 
	// Block 0x5, offset 0x140
	0x0140:0x0000000c, 0x0141:0x0000000d, 0x0142:0x0000000e, 
	0x0150:0x0000000f, 
	0x0155:0x00000010, 
	// Block 0x6, offset 0x180
	0x01bf:0x00000005, 
	// Block 0x7, offset 0x1c0
	0x01c0:0x00000006, 
	// Block 0x8, offset 0x200
	0x0219:0x00000007, 
	// Block 0x9, offset 0x240
	0x027f:0x00000008, 
	// Block 0xa, offset 0x280
	0x0280:0x00000009, 
	// Block 0xb, offset 0x2c0
	0x02c1:0x0000000a, 
	// Block 0xc, offset 0x300
	0x033f:0x0000000b, 
}

// testLookup: 640 entries, 1280 bytes
// Block 0 is the null block.
var testLookup = [640]uint16 {
	// Block 0x0, offset 0x0
	// Block 0x1, offset 0x40
	// Block 0x2, offset 0x80
	// Block 0x3, offset 0xc0
	0x0e0:0x05, 0x0e6:0x06, 
	// Block 0x4, offset 0x100
	0x13f:0x07, 
	// Block 0x5, offset 0x140
	0x140:0x08, 0x144:0x09, 
	// Block 0x6, offset 0x180
	0x190:0x03, 
	// Block 0x7, offset 0x1c0
	0x1ff:0x0a, 
	// Block 0x8, offset 0x200
	0x20f:0x05, 
	// Block 0x9, offset 0x240
	0x242:0x01, 0x244:0x02, 
	0x248:0x03, 
	0x25f:0x04, 
	0x260:0x01, 
	0x26f:0x02, 
	0x270:0x04, 0x274:0x06, 
}

var testTrie = trie{ testLookup[1216:], testValues[0:], testLookup[:], testValues[:]}`
