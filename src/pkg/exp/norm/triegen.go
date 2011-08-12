// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trie table generator.
// Used by make*tables tools to generate a go file with trie data structures
// for mapping UTF-8 to a 16-bit value. All but the last byte in a UTF-8 byte
// sequence are used to lookup offsets in the index table to be used for the
// next byte. The last byte is used to index into a table with 16-bit values.

package main

import (
	"fmt"
	"hash/crc32"
	"log"
	"utf8"
)

// Intermediate trie structure
type trieNode struct {
	table [256]*trieNode
	value uint16
	b     byte
	leaf  bool
}

func newNode() *trieNode {
	return new(trieNode)
}

func (n trieNode) String() string {
	s := fmt.Sprint("trieNode{table: { non-nil at index: ")
	for i, v := range n.table {
		if v != nil {
			s += fmt.Sprintf("%d, ", i)
		}
	}
	s += fmt.Sprintf("}, value:%#x, b:%#x leaf:%v}", n.value, n.b, n.leaf)
	return s
}

func (n trieNode) isInternal() bool {
	internal := true
	for i := 0; i < 256; i++ {
		if nn := n.table[i]; nn != nil {
			if !internal && !nn.leaf {
				log.Fatalf("triegen: isInternal: node contains both leaf and non-leaf children (%v)", n)
			}
			internal = internal && !nn.leaf
		}
	}
	return internal
}

func (n *trieNode) insert(rune int, value uint16) {
	var p [utf8.UTFMax]byte
	sz := utf8.EncodeRune(p[:], rune)

	for i := 0; i < sz; i++ {
		if n.leaf {
			log.Fatalf("triegen: insert: node (%#v) should not be a leaf", n)
		}
		nn := n.table[p[i]]
		if nn == nil {
			nn = newNode()
			nn.b = p[i]
			n.table[p[i]] = nn
		}
		n = nn
	}
	n.value = value
	n.leaf = true
}

type nodeIndex struct {
	lookupBlocks []*trieNode
	valueBlocks  []*trieNode

	lookupBlockIdx map[uint32]uint16
	valueBlockIdx  map[uint32]uint16
}

func newIndex() *nodeIndex {
	index := &nodeIndex{}
	index.lookupBlocks = make([]*trieNode, 0)
	index.valueBlocks = make([]*trieNode, 0)
	index.lookupBlockIdx = make(map[uint32]uint16)
	index.valueBlockIdx = make(map[uint32]uint16)
	return index
}

func computeOffsets(index *nodeIndex, n *trieNode) uint16 {
	if n.leaf {
		return n.value
	}
	hasher := crc32.New(crc32.MakeTable(crc32.IEEE))
	// We only index continuation bytes.
	for i := 0; i < 64; i++ {
		var v uint16 = 0
		if nn := n.table[0x80+i]; nn != nil {
			v = computeOffsets(index, nn)
		}
		hasher.Write([]byte{uint8(v >> 8), uint8(v)})
	}
	h := hasher.Sum32()
	if n.isInternal() {
		v, ok := index.lookupBlockIdx[h]
		if !ok {
			v = uint16(len(index.lookupBlocks))
			index.lookupBlocks = append(index.lookupBlocks, n)
			index.lookupBlockIdx[h] = v
		}
		n.value = v
	} else {
		v, ok := index.valueBlockIdx[h]
		if !ok {
			v = uint16(len(index.valueBlocks))
			index.valueBlocks = append(index.valueBlocks, n)
			index.valueBlockIdx[h] = v
		}
		n.value = v
	}
	return n.value
}

func printValueBlock(nr int, n *trieNode, offset int) {
	boff := nr * 64
	fmt.Printf("\n// Block %#x, offset %#x", nr, boff)
	var printnewline bool
	for i := 0; i < 64; i++ {
		if i%6 == 0 {
			printnewline = true
		}
		v := uint16(0)
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v != 0 {
			if printnewline {
				fmt.Printf("\n")
				printnewline = false
			}
			fmt.Printf("%#04x:%#04x, ", nr*64+i, v)
		}
	}
}

func printLookupBlock(nr int, n *trieNode, offset int) {
	boff := nr * 64
	fmt.Printf("\n// Block %#x, offset %#x", nr, boff)
	var printnewline bool
	for i := 0; i < 64; i++ {
		if i%8 == 0 {
			printnewline = true
		}
		v := uint16(0)
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v != 0 {
			if printnewline {
				fmt.Printf("\n")
				printnewline = false
			}
			fmt.Printf("%#03x:%#02x, ", boff+i, v)
		}
	}
}

// printTables returns the size in bytes of the generated tables.
func (t *trieNode) printTables(name string) int {
	index := newIndex()
	// Values for 7-bit ASCII are stored in first two block, followed by nil block.
	index.valueBlocks = append(index.valueBlocks, nil, nil, nil)
	// First byte of multi-byte UTF-8 codepoints are indexed in 4th block.
	index.lookupBlocks = append(index.lookupBlocks, nil, nil, nil, nil)
	// Index starter bytes of multi-byte UTF-8.
	for i := 0xC0; i < 0x100; i++ {
		if t.table[i] != nil {
			computeOffsets(index, t.table[i])
		}
	}

	nv := len(index.valueBlocks) * 64
	fmt.Printf("// %sValues: %d entries, %d bytes\n", name, nv, nv*2)
	fmt.Printf("// Block 2 is the null block.\n")
	fmt.Printf("var %sValues = [%d]uint16 {", name, nv)
	printValueBlock(0, t, 0)
	printValueBlock(1, t, 64)
	printValueBlock(2, newNode(), 0)
	for i := 3; i < len(index.valueBlocks); i++ {
		printValueBlock(i, index.valueBlocks[i], 0x80)
	}
	fmt.Print("\n}\n\n")

	ni := len(index.lookupBlocks) * 64
	fmt.Printf("// %sLookup: %d bytes\n", name, ni)
	fmt.Printf("// Block 0 is the null block.\n")
	fmt.Printf("var %sLookup = [%d]uint8 {", name, ni)
	printLookupBlock(0, newNode(), 0)
	printLookupBlock(1, newNode(), 0)
	printLookupBlock(2, newNode(), 0)
	printLookupBlock(3, t, 0xC0)
	for i := 4; i < len(index.lookupBlocks); i++ {
		printLookupBlock(i, index.lookupBlocks[i], 0x80)
	}
	fmt.Print("\n}\n\n")
	fmt.Printf("var %sTrie = trie{ %sLookup[:], %sValues[:] }\n\n", name, name, name)
	return nv*2 + ni
}
