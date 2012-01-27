// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"math"
	"sort"
)

type huffmanEncoder struct {
	codeBits []uint8
	code     []uint16
}

type literalNode struct {
	literal uint16
	freq    int32
}

type chain struct {
	// The sum of the leaves in this tree
	freq int32

	// The number of literals to the left of this item at this level
	leafCount int32

	// The right child of this chain in the previous level.
	up *chain
}

type levelInfo struct {
	// Our level.  for better printing
	level int32

	// The most recent chain generated for this level
	lastChain *chain

	// The frequency of the next character to add to this level
	nextCharFreq int32

	// The frequency of the next pair (from level below) to add to this level.
	// Only valid if the "needed" value of the next lower level is 0.
	nextPairFreq int32

	// The number of chains remaining to generate for this level before moving
	// up to the next level
	needed int32

	// The levelInfo for level+1
	up *levelInfo

	// The levelInfo for level-1
	down *levelInfo
}

func maxNode() literalNode { return literalNode{math.MaxUint16, math.MaxInt32} }

func newHuffmanEncoder(size int) *huffmanEncoder {
	return &huffmanEncoder{make([]uint8, size), make([]uint16, size)}
}

// Generates a HuffmanCode corresponding to the fixed literal table
func generateFixedLiteralEncoding() *huffmanEncoder {
	h := newHuffmanEncoder(maxLit)
	codeBits := h.codeBits
	code := h.code
	var ch uint16
	for ch = 0; ch < maxLit; ch++ {
		var bits uint16
		var size uint8
		switch {
		case ch < 144:
			// size 8, 000110000  .. 10111111
			bits = ch + 48
			size = 8
			break
		case ch < 256:
			// size 9, 110010000 .. 111111111
			bits = ch + 400 - 144
			size = 9
			break
		case ch < 280:
			// size 7, 0000000 .. 0010111
			bits = ch - 256
			size = 7
			break
		default:
			// size 8, 11000000 .. 11000111
			bits = ch + 192 - 280
			size = 8
		}
		codeBits[ch] = size
		code[ch] = reverseBits(bits, size)
	}
	return h
}

func generateFixedOffsetEncoding() *huffmanEncoder {
	h := newHuffmanEncoder(30)
	codeBits := h.codeBits
	code := h.code
	for ch := uint16(0); ch < 30; ch++ {
		codeBits[ch] = 5
		code[ch] = reverseBits(ch, 5)
	}
	return h
}

var fixedLiteralEncoding *huffmanEncoder = generateFixedLiteralEncoding()
var fixedOffsetEncoding *huffmanEncoder = generateFixedOffsetEncoding()

func (h *huffmanEncoder) bitLength(freq []int32) int64 {
	var total int64
	for i, f := range freq {
		if f != 0 {
			total += int64(f) * int64(h.codeBits[i])
		}
	}
	return total
}

// Return the number of literals assigned to each bit size in the Huffman encoding
//
// This method is only called when list.length >= 3
// The cases of 0, 1, and 2 literals are handled by special case code.
//
// list  An array of the literals with non-zero frequencies
//             and their associated frequencies.  The array is in order of increasing
//             frequency, and has as its last element a special element with frequency
//             MaxInt32
// maxBits     The maximum number of bits that should be used to encode any literal.
// return      An integer array in which array[i] indicates the number of literals
//             that should be encoded in i bits.
func (h *huffmanEncoder) bitCounts(list []literalNode, maxBits int32) []int32 {
	n := int32(len(list))
	list = list[0 : n+1]
	list[n] = maxNode()

	// The tree can't have greater depth than n - 1, no matter what.  This
	// saves a little bit of work in some small cases
	if maxBits > n-1 {
		maxBits = n - 1
	}

	// Create information about each of the levels.
	// A bogus "Level 0" whose sole purpose is so that
	// level1.prev.needed==0.  This makes level1.nextPairFreq
	// be a legitimate value that never gets chosen.
	top := &levelInfo{needed: 0}
	chain2 := &chain{list[1].freq, 2, new(chain)}
	for level := int32(1); level <= maxBits; level++ {
		// For every level, the first two items are the first two characters.
		// We initialize the levels as if we had already figured this out.
		top = &levelInfo{
			level:        level,
			lastChain:    chain2,
			nextCharFreq: list[2].freq,
			nextPairFreq: list[0].freq + list[1].freq,
			down:         top,
		}
		top.down.up = top
		if level == 1 {
			top.nextPairFreq = math.MaxInt32
		}
	}

	// We need a total of 2*n - 2 items at top level and have already generated 2.
	top.needed = 2*n - 4

	l := top
	for {
		if l.nextPairFreq == math.MaxInt32 && l.nextCharFreq == math.MaxInt32 {
			// We've run out of both leafs and pairs.
			// End all calculations for this level.
			// To m sure we never come back to this level or any lower level,
			// set nextPairFreq impossibly large.
			l.lastChain = nil
			l.needed = 0
			l = l.up
			l.nextPairFreq = math.MaxInt32
			continue
		}

		prevFreq := l.lastChain.freq
		if l.nextCharFreq < l.nextPairFreq {
			// The next item on this row is a leaf node.
			n := l.lastChain.leafCount + 1
			l.lastChain = &chain{l.nextCharFreq, n, l.lastChain.up}
			l.nextCharFreq = list[n].freq
		} else {
			// The next item on this row is a pair from the previous row.
			// nextPairFreq isn't valid until we generate two
			// more values in the level below
			l.lastChain = &chain{l.nextPairFreq, l.lastChain.leafCount, l.down.lastChain}
			l.down.needed = 2
		}

		if l.needed--; l.needed == 0 {
			// We've done everything we need to do for this level.
			// Continue calculating one level up.  Fill in nextPairFreq
			// of that level with the sum of the two nodes we've just calculated on
			// this level.
			up := l.up
			if up == nil {
				// All done!
				break
			}
			up.nextPairFreq = prevFreq + l.lastChain.freq
			l = up
		} else {
			// If we stole from below, move down temporarily to replenish it.
			for l.down.needed > 0 {
				l = l.down
			}
		}
	}

	// Somethings is wrong if at the end, the top level is null or hasn't used
	// all of the leaves.
	if top.lastChain.leafCount != n {
		panic("top.lastChain.leafCount != n")
	}

	bitCount := make([]int32, maxBits+1)
	bits := 1
	for chain := top.lastChain; chain.up != nil; chain = chain.up {
		// chain.leafCount gives the number of literals requiring at least "bits"
		// bits to encode.
		bitCount[bits] = chain.leafCount - chain.up.leafCount
		bits++
	}
	return bitCount
}

// Look at the leaves and assign them a bit count and an encoding as specified
// in RFC 1951 3.2.2
func (h *huffmanEncoder) assignEncodingAndSize(bitCount []int32, list []literalNode) {
	code := uint16(0)
	for n, bits := range bitCount {
		code <<= 1
		if n == 0 || bits == 0 {
			continue
		}
		// The literals list[len(list)-bits] .. list[len(list)-bits]
		// are encoded using "bits" bits, and get the values
		// code, code + 1, ....  The code values are
		// assigned in literal order (not frequency order).
		chunk := list[len(list)-int(bits):]
		sortByLiteral(chunk)
		for _, node := range chunk {
			h.codeBits[node.literal] = uint8(n)
			h.code[node.literal] = reverseBits(code, uint8(n))
			code++
		}
		list = list[0 : len(list)-int(bits)]
	}
}

// Update this Huffman Code object to be the minimum code for the specified frequency count.
//
// freq  An array of frequencies, in which frequency[i] gives the frequency of literal i.
// maxBits  The maximum number of bits to use for any literal.
func (h *huffmanEncoder) generate(freq []int32, maxBits int32) {
	list := make([]literalNode, len(freq)+1)
	// Number of non-zero literals
	count := 0
	// Set list to be the set of all non-zero literals and their frequencies
	for i, f := range freq {
		if f != 0 {
			list[count] = literalNode{uint16(i), f}
			count++
		} else {
			h.codeBits[i] = 0
		}
	}
	// If freq[] is shorter than codeBits[], fill rest of codeBits[] with zeros
	h.codeBits = h.codeBits[0:len(freq)]
	list = list[0:count]
	if count <= 2 {
		// Handle the small cases here, because they are awkward for the general case code.  With
		// two or fewer literals, everything has bit length 1.
		for i, node := range list {
			// "list" is in order of increasing literal value.
			h.codeBits[node.literal] = 1
			h.code[node.literal] = uint16(i)
		}
		return
	}
	sortByFreq(list)

	// Get the number of literals for each bit count
	bitCount := h.bitCounts(list, maxBits)
	// And do the assignment
	h.assignEncodingAndSize(bitCount, list)
}

type literalNodeSorter struct {
	a    []literalNode
	less func(i, j int) bool
}

func (s literalNodeSorter) Len() int { return len(s.a) }

func (s literalNodeSorter) Less(i, j int) bool {
	return s.less(i, j)
}

func (s literalNodeSorter) Swap(i, j int) { s.a[i], s.a[j] = s.a[j], s.a[i] }

func sortByFreq(a []literalNode) {
	s := &literalNodeSorter{a, func(i, j int) bool {
		if a[i].freq == a[j].freq {
			return a[i].literal < a[j].literal
		}
		return a[i].freq < a[j].freq
	}}
	sort.Sort(s)
}

func sortByLiteral(a []literalNode) {
	s := &literalNodeSorter{a, func(i, j int) bool { return a[i].literal < a[j].literal }}
	sort.Sort(s)
}
