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

const blockSize = 64
const maxSparseEntries = 16

// Intermediate trie structure
type trieNode struct {
	table [256]*trieNode
	value int
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

func (n trieNode) mostFrequentStride() int {
	counts := make(map[int]int)
	v := 0
	for _, t := range n.table[0x80 : 0x80+blockSize] {
		if t != nil {
			if stride := t.value - v; v != 0 && stride >= 0 {
				counts[stride]++
			}
			v = t.value
		}
	}
	var maxs, maxc int
	for stride, cnt := range counts {
		if cnt > maxc {
			maxs, maxc = stride, cnt
		}
	}
	return maxs
}

func (n trieNode) countSparseEntries() int {
	stride := n.mostFrequentStride()
	var count, v int
	for _, t := range n.table[0x80 : 0x80+blockSize] {
		tv := 0
		if t != nil {
			tv = t.value
		}
		if tv-v != stride {
			if tv != 0 {
				count++
			}
		}
		v = tv
	}
	return count
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
	n.value = int(value)
	n.leaf = true
}

type nodeIndex struct {
	lookupBlocks []*trieNode
	valueBlocks  []*trieNode
	sparseBlocks []*trieNode
	sparseOffset []uint16
	sparseCount  int

	lookupBlockIdx map[uint32]int
	valueBlockIdx  map[uint32]int
}

func newIndex() *nodeIndex {
	index := &nodeIndex{}
	index.lookupBlocks = make([]*trieNode, 0)
	index.valueBlocks = make([]*trieNode, 0)
	index.sparseBlocks = make([]*trieNode, 0)
	index.sparseOffset = make([]uint16, 1)
	index.lookupBlockIdx = make(map[uint32]int)
	index.valueBlockIdx = make(map[uint32]int)
	return index
}

func computeOffsets(index *nodeIndex, n *trieNode) int {
	if n.leaf {
		return n.value
	}
	hasher := crc32.New(crc32.MakeTable(crc32.IEEE))
	// We only index continuation bytes.
	for i := 0; i < blockSize; i++ {
		v := 0
		if nn := n.table[0x80+i]; nn != nil {
			v = computeOffsets(index, nn)
		}
		hasher.Write([]byte{uint8(v >> 8), uint8(v)})
	}
	h := hasher.Sum32()
	if n.isInternal() {
		v, ok := index.lookupBlockIdx[h]
		if !ok {
			v = len(index.lookupBlocks)
			index.lookupBlocks = append(index.lookupBlocks, n)
			index.lookupBlockIdx[h] = v
		}
		n.value = v
	} else {
		v, ok := index.valueBlockIdx[h]
		if !ok {
			if c := n.countSparseEntries(); c > maxSparseEntries {
				v = len(index.valueBlocks)
				index.valueBlocks = append(index.valueBlocks, n)
				index.valueBlockIdx[h] = v
			} else {
				v = -len(index.sparseOffset)
				index.sparseBlocks = append(index.sparseBlocks, n)
				index.sparseOffset = append(index.sparseOffset, uint16(index.sparseCount))
				index.sparseCount += c + 1
				index.valueBlockIdx[h] = v
			}
		}
		n.value = v
	}
	return n.value
}

func printValueBlock(nr int, n *trieNode, offset int) {
	boff := nr * blockSize
	fmt.Printf("\n// Block %#x, offset %#x", nr, boff)
	var printnewline bool
	for i := 0; i < blockSize; i++ {
		if i%6 == 0 {
			printnewline = true
		}
		v := 0
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v != 0 {
			if printnewline {
				fmt.Printf("\n")
				printnewline = false
			}
			fmt.Printf("%#04x:%#04x, ", boff+i, v)
		}
	}
}

func printSparseBlock(nr int, n *trieNode) {
	boff := -n.value
	fmt.Printf("\n// Block %#x, offset %#x", nr, boff)
	v := 0
	//stride := f(n)
	stride := n.mostFrequentStride()
	c := n.countSparseEntries()
	fmt.Printf("\n{value:%#04x,lo:%#02x},", stride, uint8(c))
	for i, nn := range n.table[0x80 : 0x80+blockSize] {
		nv := 0
		if nn != nil {
			nv = nn.value
		}
		if nv-v != stride {
			if v != 0 {
				fmt.Printf(",hi:%#02x},", 0x80+i-1)
			}
			if nv != 0 {
				fmt.Printf("\n{value:%#04x,lo:%#02x", nv, nn.b)
			}
		}
		v = nv
	}
	if v != 0 {
		fmt.Printf(",hi:%#02x},", 0x80+blockSize-1)
	}
}

func printLookupBlock(nr int, n *trieNode, offset, cutoff int) {
	boff := nr * blockSize
	fmt.Printf("\n// Block %#x, offset %#x", nr, boff)
	var printnewline bool
	for i := 0; i < blockSize; i++ {
		if i%8 == 0 {
			printnewline = true
		}
		v := 0
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v != 0 {
			if v < 0 {
				v = -v - 1 + cutoff
			}
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

	nv := len(index.valueBlocks) * blockSize
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

	ls := len(index.sparseBlocks)
	fmt.Printf("// %sSparseOffset: %d entries, %d bytes\n", name, ls, ls*2)
	fmt.Printf("var %sSparseOffset = %#v\n\n", name, index.sparseOffset[1:])

	ns := index.sparseCount
	fmt.Printf("// %sSparseValues: %d entries, %d bytes\n", name, ns, ns*4)
	fmt.Printf("var %sSparseValues = [%d]valueRange {", name, ns)
	for i, n := range index.sparseBlocks {
		printSparseBlock(i, n)
	}
	fmt.Print("\n}\n\n")

	cutoff := len(index.valueBlocks)
	ni := len(index.lookupBlocks) * blockSize
	fmt.Printf("// %sLookup: %d bytes\n", name, ni)
	fmt.Printf("// Block 0 is the null block.\n")
	fmt.Printf("var %sLookup = [%d]uint8 {", name, ni)
	printLookupBlock(0, newNode(), 0, cutoff)
	printLookupBlock(1, newNode(), 0, cutoff)
	printLookupBlock(2, newNode(), 0, cutoff)
	printLookupBlock(3, t, 0xC0, cutoff)
	for i := 4; i < len(index.lookupBlocks); i++ {
		printLookupBlock(i, index.lookupBlocks[i], 0x80, cutoff)
	}
	fmt.Print("\n}\n\n")
	fmt.Printf("var %sTrie = trie{ %sLookup[:], %sValues[:], %sSparseValues[:], %sSparseOffset[:], %d}\n\n",
		name, name, name, name, name, cutoff)
	return nv*2 + ns*4 + ni + ls*2
}
