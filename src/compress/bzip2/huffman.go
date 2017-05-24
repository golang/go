// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bzip2

import "sort"

// A huffmanTree is a binary tree which is navigated, bit-by-bit to reach a
// symbol.
type huffmanTree struct {
	// nodes contains all the non-leaf nodes in the tree. nodes[0] is the
	// root of the tree and nextNode contains the index of the next element
	// of nodes to use when the tree is being constructed.
	nodes    []huffmanNode
	nextNode int
}

// A huffmanNode is a node in the tree. left and right contain indexes into the
// nodes slice of the tree. If left or right is invalidNodeValue then the child
// is a left node and its value is in leftValue/rightValue.
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
			bit = 0 - (uint16(br.n>>br.bits) & 1)
		} else {
			// Get next bit - slow path.
			// Use ReadBits to retrieve a single bit
			// from the underling io.ByteReader.
			bit = 0 - uint16(br.ReadBits(1))
		}
		// now
		// bit = 0xffff if the next bit was 1
		// bit = 0x0000 if the next bit was 0

		// 1 means left, 0 means right.
		//
		// if bit == 0xffff {
		//     nodeIndex = node.left
		// } else {
		//     nodeIndex = node.right
		// }
		nodeIndex = (bit & node.left) | (^bit & node.right)

		if nodeIndex == invalidNodeValue {
			// We found a leaf. Use the value of bit to decide
			// whether is a left or a right value.
			return (bit & node.leftValue) | (^bit & node.rightValue)
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
	pairs := huffmanSymbolLengthPairs(make([]huffmanSymbolLengthPair, len(lengths)))
	for i, length := range lengths {
		pairs[i].value = uint16(i)
		pairs[i].length = length
	}

	sort.Sort(pairs)

	// Now we assign codes to the symbols, starting with the longest code.
	// We keep the codes packed into a uint32, at the most-significant end.
	// So branches are taken from the MSB downwards. This makes it easy to
	// sort them later.
	code := uint32(0)
	length := uint8(32)

	codes := huffmanCodes(make([]huffmanCode, len(lengths)))
	for i := len(pairs) - 1; i >= 0; i-- {
		if length > pairs[i].length {
			length = pairs[i].length
		}
		codes[i].code = code
		codes[i].codeLen = length
		codes[i].value = pairs[i].value
		// We need to 'increment' the code, which means treating |code|
		// like a |length| bit number.
		code += 1 << (32 - length)
	}

	// Now we can sort by the code so that the left half of each branch are
	// grouped together, recursively.
	sort.Sort(codes)

	t.nodes = make([]huffmanNode, len(codes))
	_, err := buildHuffmanNode(&t, codes, 0)
	return t, err
}

// huffmanSymbolLengthPair contains a symbol and its code length.
type huffmanSymbolLengthPair struct {
	value  uint16
	length uint8
}

// huffmanSymbolLengthPair is used to provide an interface for sorting.
type huffmanSymbolLengthPairs []huffmanSymbolLengthPair

func (h huffmanSymbolLengthPairs) Len() int {
	return len(h)
}

func (h huffmanSymbolLengthPairs) Less(i, j int) bool {
	if h[i].length < h[j].length {
		return true
	}
	if h[i].length > h[j].length {
		return false
	}
	if h[i].value < h[j].value {
		return true
	}
	return false
}

func (h huffmanSymbolLengthPairs) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

// huffmanCode contains a symbol, its code and code length.
type huffmanCode struct {
	code    uint32
	codeLen uint8
	value   uint16
}

// huffmanCodes is used to provide an interface for sorting.
type huffmanCodes []huffmanCode

func (n huffmanCodes) Len() int {
	return len(n)
}

func (n huffmanCodes) Less(i, j int) bool {
	return n[i].code < n[j].code
}

func (n huffmanCodes) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
}

// buildHuffmanNode takes a slice of sorted huffmanCodes and builds a node in
// the Huffman tree at the given level. It returns the index of the newly
// constructed node.
func buildHuffmanNode(t *huffmanTree, codes []huffmanCode, level uint32) (nodeIndex uint16, err error) {
	test := uint32(1) << (31 - level)

	// We have to search the list of codes to find the divide between the left and right sides.
	firstRightIndex := len(codes)
	for i, code := range codes {
		if code.code&test != 0 {
			firstRightIndex = i
			break
		}
	}

	left := codes[:firstRightIndex]
	right := codes[firstRightIndex:]

	if len(left) == 0 || len(right) == 0 {
		// There is a superfluous level in the Huffman tree indicating
		// a bug in the encoder. However, this bug has been observed in
		// the wild so we handle it.

		// If this function was called recursively then we know that
		// len(codes) >= 2 because, otherwise, we would have hit the
		// "leaf node" case, below, and not recursed.
		//
		// However, for the initial call it's possible that len(codes)
		// is zero or one. Both cases are invalid because a zero length
		// tree cannot encode anything and a length-1 tree can only
		// encode EOF and so is superfluous. We reject both.
		if len(codes) < 2 {
			return 0, StructuralError("empty Huffman tree")
		}

		// In this case the recursion doesn't always reduce the length
		// of codes so we need to ensure termination via another
		// mechanism.
		if level == 31 {
			// Since len(codes) >= 2 the only way that the values
			// can match at all 32 bits is if they are equal, which
			// is invalid. This ensures that we never enter
			// infinite recursion.
			return 0, StructuralError("equal symbols in Huffman tree")
		}

		if len(left) == 0 {
			return buildHuffmanNode(t, right, level+1)
		}
		return buildHuffmanNode(t, left, level+1)
	}

	nodeIndex = uint16(t.nextNode)
	node := &t.nodes[t.nextNode]
	t.nextNode++

	if len(left) == 1 {
		// leaf node
		node.left = invalidNodeValue
		node.leftValue = left[0].value
	} else {
		node.left, err = buildHuffmanNode(t, left, level+1)
	}

	if err != nil {
		return
	}

	if len(right) == 1 {
		// leaf node
		node.right = invalidNodeValue
		node.rightValue = right[0].value
	} else {
		node.right, err = buildHuffmanNode(t, right, level+1)
	}

	return
}
