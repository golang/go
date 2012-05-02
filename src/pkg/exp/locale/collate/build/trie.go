// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The trie in this file is used to associate the first full character
// in a UTF-8 string to a collation element.
// All but the last byte in a UTF-8 byte sequence are 
// used to look up offsets in the index table to be used for the next byte.
// The last byte is used to index into a table of collation elements.
// This file contains the code for the generation of the trie.

package build

import (
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"reflect"
)

const (
	blockSize   = 64
	blockOffset = 2 // Substract 2 blocks to compensate for the 0x80 added to continuation bytes.
)

type trie struct {
	index  []uint16
	values []uint32
}

// trieNode is the intermediate trie structure used for generating a trie.
type trieNode struct {
	table [256]*trieNode
	value int64
	b     byte
	leaf  bool
}

func newNode() *trieNode {
	return new(trieNode)
}

func (n *trieNode) isInternal() bool {
	internal := true
	for i := 0; i < 256; i++ {
		if nn := n.table[i]; nn != nil {
			if !internal && !nn.leaf {
				log.Fatalf("trie:isInternal: node contains both leaf and non-leaf children (%v)", n)
			}
			internal = internal && !nn.leaf
		}
	}
	return internal
}

func (n *trieNode) insert(r rune, value uint32) {
	for _, b := range []byte(string(r)) {
		if n.leaf {
			log.Fatalf("trie:insert: node (%#v) should not be a leaf", n)
		}
		nn := n.table[b]
		if nn == nil {
			nn = newNode()
			nn.b = b
			n.table[b] = nn
		}
		n = nn
	}
	n.value = int64(value)
	n.leaf = true
}

type nodeIndex struct {
	lookupBlocks []*trieNode
	valueBlocks  []*trieNode

	lookupBlockIdx map[uint32]int64
	valueBlockIdx  map[uint32]int64
}

func newIndex() *nodeIndex {
	index := &nodeIndex{}
	index.lookupBlocks = make([]*trieNode, 0)
	index.valueBlocks = make([]*trieNode, 0)
	index.lookupBlockIdx = make(map[uint32]int64)
	index.valueBlockIdx = make(map[uint32]int64)
	return index
}

func computeOffsets(index *nodeIndex, n *trieNode) int64 {
	if n.leaf {
		return n.value
	}
	hasher := fnv.New32()
	// We only index continuation bytes.
	for i := 0; i < blockSize; i++ {
		v := int64(0)
		if nn := n.table[0x80+i]; nn != nil {
			v = computeOffsets(index, nn)
		}
		hasher.Write([]byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)})
	}
	h := hasher.Sum32()
	if n.isInternal() {
		v, ok := index.lookupBlockIdx[h]
		if !ok {
			v = int64(len(index.lookupBlocks)) - blockOffset
			index.lookupBlocks = append(index.lookupBlocks, n)
			index.lookupBlockIdx[h] = v
		}
		n.value = v
	} else {
		v, ok := index.valueBlockIdx[h]
		if !ok {
			v = int64(len(index.valueBlocks)) - blockOffset
			index.valueBlocks = append(index.valueBlocks, n)
			index.valueBlockIdx[h] = v
		}
		n.value = v
	}
	return n.value
}

func genValueBlock(t *trie, n *trieNode, offset int) error {
	for i := 0; i < blockSize; i++ {
		v := int64(0)
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v >= 1<<32 {
			return fmt.Errorf("value %d at index %d does not fit in uint32", v, len(t.values))
		}
		t.values = append(t.values, uint32(v))
	}
	return nil
}

func genLookupBlock(t *trie, n *trieNode, offset int) error {
	for i := 0; i < blockSize; i++ {
		v := int64(0)
		if nn := n.table[i+offset]; nn != nil {
			v = nn.value
		}
		if v >= 1<<16 {
			return fmt.Errorf("value %d at index %d does not fit in uint16", v, len(t.index))
		}
		t.index = append(t.index, uint16(v))
	}
	return nil
}

// generate generates and returns the trie for n.
func (n *trieNode) generate() (t *trie, err error) {
	seterr := func(e error) {
		if err == nil {
			err = e
		}
	}
	index := newIndex()
	// Values for 7-bit ASCII are stored in the first of two blocks, followed by a nil block.
	index.valueBlocks = append(index.valueBlocks, nil, nil, nil)
	// First byte of multi-byte UTF-8 codepoints are indexed in 4th block.
	index.lookupBlocks = append(index.lookupBlocks, nil, nil, nil, nil)
	// Index starter bytes of multi-byte UTF-8.
	for i := 0xC0; i < 0x100; i++ {
		if n.table[i] != nil {
			computeOffsets(index, n.table[i])
		}
	}
	t = &trie{}
	seterr(genValueBlock(t, n, 0))
	seterr(genValueBlock(t, n, 64))
	seterr(genValueBlock(t, newNode(), 0))
	for i := 3; i < len(index.valueBlocks); i++ {
		seterr(genValueBlock(t, index.valueBlocks[i], 0x80))
	}
	if len(index.valueBlocks) >= 1<<16 {
		seterr(fmt.Errorf("maximum number of value blocks exceeded (%d > %d)", len(index.valueBlocks), 1<<16))
		return
	}
	seterr(genLookupBlock(t, newNode(), 0))
	seterr(genLookupBlock(t, newNode(), 0))
	seterr(genLookupBlock(t, newNode(), 0))
	seterr(genLookupBlock(t, n, 0xC0))
	for i := 4; i < len(index.lookupBlocks); i++ {
		seterr(genLookupBlock(t, index.lookupBlocks[i], 0x80))
	}
	return
}

// print writes a compilable trie to w.  It returns the number of characters
// printed and the size of the generated structure in bytes.
func (t *trie) print(w io.Writer, name string) (n, size int, err error) {
	update3 := func(nn, sz int, e error) {
		n += nn
		if err == nil {
			err = e
		}
		size += sz
	}
	update2 := func(nn int, e error) { update3(nn, 0, e) }

	update3(t.printArrays(w, name))
	update2(fmt.Fprintf(w, "var %sTrie = ", name))
	update3(t.printStruct(w, name))
	update2(fmt.Fprintln(w))
	return
}

func (t *trie) printArrays(w io.Writer, name string) (n, size int, err error) {
	p := func(f string, a ...interface{}) {
		nn, e := fmt.Fprintf(w, f, a...)
		n += nn
		if err == nil {
			err = e
		}
	}
	nv := len(t.values)
	p("// %sValues: %d entries, %d bytes\n", name, nv, nv*4)
	p("// Block 2 is the null block.\n")
	p("var %sValues = [%d]uint32 {", name, nv)
	var printnewline bool
	for i, v := range t.values {
		if i%blockSize == 0 {
			p("\n\t// Block %#x, offset %#x", i/blockSize, i)
		}
		if i%4 == 0 {
			printnewline = true
		}
		if v != 0 {
			if printnewline {
				p("\n\t")
				printnewline = false
			}
			p("%#04x:%#08x, ", i, v)
		}
	}
	p("\n}\n\n")
	ni := len(t.index)
	p("// %sLookup: %d entries, %d bytes\n", name, ni, ni*2)
	p("// Block 0 is the null block.\n")
	p("var %sLookup = [%d]uint16 {", name, ni)
	printnewline = false
	for i, v := range t.index {
		if i%blockSize == 0 {
			p("\n\t// Block %#x, offset %#x", i/blockSize, i)
		}
		if i%8 == 0 {
			printnewline = true
		}
		if v != 0 {
			if printnewline {
				p("\n\t")
				printnewline = false
			}
			p("%#03x:%#02x, ", i, v)
		}
	}
	p("\n}\n\n")
	return n, nv*4 + ni*2, err
}

func (t *trie) printStruct(w io.Writer, name string) (n, sz int, err error) {
	n, err = fmt.Fprintf(w, "trie{ %sLookup[:], %sValues[:]}", name, name)
	sz += int(reflect.TypeOf(trie{}).Size())
	return
}
