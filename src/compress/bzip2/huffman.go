// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bzip2

import (
	"cmp"
	"slices"
)

// A huffmanTree is a binary tree which is navigated, bit-by-bit to reach a
// symbol.
type huffmanTree struct {
	// nodes contains all the non-leaf nodes. nodes[0] is the root.
	nodes []huffmanNode
}

// A huffmanNode is a node in the tree. left and right contain indexes into the
// nodes slice of the tree. If left or right is invalidNodeValue then the child
// is a leaf and its value is in leftValue/rightValue. An unused branch
// is represented by a leaf with value invalidHuffmanSymbol.
//
// The symbols are uint16s because bzip2 encodes not only MTF indexes in the
// tree, but also two magic values for run-length encoding and an EOF symbol.
// Thus there are more than 256 possible symbols.
type huffmanNode struct {
	left, right           uint16
	leftValue, rightValue uint16
}

// invalidNodeValue is an invalid index which marks a leaf node in the tree.
const invalidNodeValue = 0xffff
const invalidHuffmanSymbol = 0xffff

// Decode reads bits from the given bitReader and navigates the tree until a
// symbol is found.
func (t *huffmanTree) Decode(br *bitReader) (v uint16) {
	nodeIndex := uint16(0) // node 0 is the root of the tree.

	for {
		node := &t.nodes[nodeIndex]

		var bit uint16
		if br.bits > 0 {
			// Get next bit - fast path.
			br.bits--
			bit = uint16(br.n>>(br.bits&63)) & 1
		} else {
			// Get next bit - slow path.
			// Use ReadBits to retrieve a single bit
			// from the underling io.ByteReader.
			bit = uint16(br.ReadBits(1))
			if br.err != nil {
				return invalidHuffmanSymbol
			}
		}

		// Trick a compiler into generating conditional move instead of branch,
		// by making both loads unconditional.
		l, r := node.left, node.right

		if bit == 1 {
			nodeIndex = l
		} else {
			nodeIndex = r
		}

		if nodeIndex == invalidNodeValue {
			// We found a leaf. Use the value of bit to decide
			// whether is a left or a right value.
			l, r := node.leftValue, node.rightValue
			if bit == 1 {
				v = l
			} else {
				v = r
			}
			return
		}
	}
}

// newHuffmanTree builds a Huffman tree from a slice containing the code
// lengths of each symbol. The maximum code length is 32 bits.
func newHuffmanTree(lengths []uint8) (huffmanTree, error) {
	// There are many possible trees that assign the same code length to
	// each symbol (consider reflecting a tree down the middle, for
	// example). Since the code length assignments determine the
	// efficiency of the tree, each of these trees is equally good. In
	// order to minimize the amount of information needed to build a tree
	// bzip2 uses a canonical tree so that it can be reconstructed given
	// only the code length assignments.

	if len(lengths) < 2 {
		panic("newHuffmanTree: too few symbols")
	}

	var t huffmanTree

	// First we sort the code length assignments by ascending code length,
	// using the symbol value to break ties.
	pairs := make([]huffmanSymbolLengthPair, len(lengths))
	for i, length := range lengths {
		pairs[i].value = uint16(i)
		pairs[i].length = length
	}

	slices.SortFunc(pairs, func(a, b huffmanSymbolLengthPair) int {
		if c := cmp.Compare(a.length, b.length); c != 0 {
			return c
		}
		return cmp.Compare(a.value, b.value)
	})

	// Assign canonical codes from shortest to longest. Keep the bits at
	// the high end so the codes are already sorted for tree construction.
	var code uint64
	codes := make([]huffmanCode, len(pairs))
	for i, pair := range pairs {
		if code >= 1<<32 {
			return huffmanTree{}, StructuralError("oversubscribed Huffman tree")
		}
		codes[i] = huffmanCode{uint32(code), pair.length, pair.value}
		code += uint64(1) << (32 - pair.length)
	}

	t.nodes = make([]huffmanNode, 0, len(codes))
	t.buildNode(codes, 0)
	return t, nil
}

// huffmanSymbolLengthPair contains a symbol and its code length.
type huffmanSymbolLengthPair struct {
	value  uint16
	length uint8
}

// huffmanCode contains a symbol, its code and code length.
type huffmanCode struct {
	code    uint32
	codeLen uint8
	value   uint16
}

// buildNode builds an internal node from codes that share a prefix of
// length level and returns its index.
func (t *huffmanTree) buildNode(codes []huffmanCode, level uint8) uint16 {
	index := uint16(len(t.nodes))
	t.nodes = append(t.nodes, huffmanNode{})

	bit := uint32(1) << (31 - level)
	split := len(codes)
	for i, code := range codes {
		if code.code&bit != 0 {
			split = i
			break
		}
	}

	// Decode uses the left child for a 1 bit and the right child for 0.
	left, leftValue := t.buildChild(codes[split:], level+1)
	right, rightValue := t.buildChild(codes[:split], level+1)
	t.nodes[index] = huffmanNode{left, right, leftValue, rightValue}
	return index
}

func (t *huffmanTree) buildChild(codes []huffmanCode, level uint8) (index, value uint16) {
	if len(codes) == 0 {
		return invalidNodeValue, invalidHuffmanSymbol
	}
	if len(codes) == 1 && codes[0].codeLen == level {
		return invalidNodeValue, codes[0].value
	}
	return t.buildNode(codes, level), 0
}
