// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"cmp"
	"math"
	"math/bits"
	"slices"
)

// hcode is a huffman code with a bit code and bit length.
type hcode struct {
	code, len uint16
}

type huffmanEncoder struct {
	codes     []hcode
	freqcache []literalNode
	bitCount  [17]int32
}

type literalNode struct {
	literal uint16
	freq    int32
}

// A levelInfo describes the state of the constructed tree for a given depth.
type levelInfo struct {
	// Our level.  for better printing
	level int32

	// The frequency of the last node at this level
	lastFreq int32

	// The frequency of the next character to add to this level
	nextCharFreq int32

	// The frequency of the next pair (from level below) to add to this level.
	// Only valid if the "needed" value of the next lower level is 0.
	nextPairFreq int32

	// The number of chains remaining to generate for this level before moving
	// up to the next level
	needed int32
}

// set sets the code and length of an hcode.
func (h *hcode) set(code uint16, length uint16) {
	h.len = length
	h.code = code
}

func maxNode() literalNode { return literalNode{math.MaxUint16, math.MaxInt32} }

func newHuffmanEncoder(size int) *huffmanEncoder {
	return &huffmanEncoder{codes: make([]hcode, size)}
}

// Generates a HuffmanCode corresponding to the fixed literal table.
func generateFixedLiteralEncoding() *huffmanEncoder {
	h := newHuffmanEncoder(maxNumLit)
	codes := h.codes
	var ch uint16
	for ch = 0; ch < maxNumLit; ch++ {
		var bits uint16
		var size uint16
		switch {
		case ch < 144:
			// size 8, 000110000  .. 10111111
			bits = ch + 48
			size = 8
		case ch < 256:
			// size 9, 110010000 .. 111111111
			bits = ch + 400 - 144
			size = 9
		case ch < 280:
			// size 7, 0000000 .. 0010111
			bits = ch - 256
			size = 7
		default:
			// size 8, 11000000 .. 11000111
			bits = ch + 192 - 280
			size = 8
		}
		codes[ch] = hcode{code: reverseBits(bits, byte(size)), len: size}
	}
	return h
}

func generateFixedOffsetEncoding() *huffmanEncoder {
	h := newHuffmanEncoder(30)
	codes := h.codes
	for ch := range codes {
		codes[ch] = hcode{code: reverseBits(uint16(ch), 5), len: 5}
	}
	return h
}

var fixedLiteralEncoding *huffmanEncoder = generateFixedLiteralEncoding()
var fixedOffsetEncoding *huffmanEncoder = generateFixedOffsetEncoding()

func (h *huffmanEncoder) bitLength(freq []int32) int {
	var total int
	for i, f := range freq {
		if f != 0 {
			total += int(f) * int(h.codes[i].len)
		}
	}
	return total
}

const maxBitsLimit = 16

// bitCounts computes the number of literals assigned to each bit size in the Huffman encoding.
// It is only called when list.length >= 3.
// The cases of 0, 1, and 2 literals are handled by special case code.
//
// list is an array of the literals with non-zero frequencies
// and their associated frequencies. The array is in order of increasing
// frequency and has as its last element a special element with frequency
// MaxInt32.
//
// maxBits is the maximum number of bits that should be used to encode any literal.
// It must be less than 16.
//
// bitCounts returns an integer slice in which slice[i] indicates the number of literals
// that should be encoded in i bits.
func (h *huffmanEncoder) bitCounts(list []literalNode, maxBits int32) []int32 {
	if maxBits >= maxBitsLimit {
		panic("flate: maxBits too large")
	}
	n := int32(len(list))
	list = list[0 : n+1]
	list[n] = maxNode()

	// The tree can't have greater depth than n - 1, no matter what. This
	// saves a little bit of work in some small cases
	if maxBits > n-1 {
		maxBits = n - 1
	}

	// Create information about each of the levels.
	// A bogus "Level 0" whose sole purpose is so that
	// level1.prev.needed==0.  This makes level1.nextPairFreq
	// be a legitimate value that never gets chosen.
	var levels [maxBitsLimit]levelInfo
	// leafCounts[i] counts the number of literals at the left
	// of ancestors of the rightmost node at level i.
	// leafCounts[i][j] is the number of literals at the left
	// of the level j ancestor.
	var leafCounts [maxBitsLimit][maxBitsLimit]int32

	for level := int32(1); level <= maxBits; level++ {
		// For every level, the first two items are the first two characters.
		// We initialize the levels as if we had already figured this out.
		levels[level] = levelInfo{
			level:        level,
			lastFreq:     list[1].freq,
			nextCharFreq: list[2].freq,
			nextPairFreq: list[0].freq + list[1].freq,
		}
		leafCounts[level][level] = 2
		if level == 1 {
			levels[level].nextPairFreq = math.MaxInt32
		}
	}

	// We need a total of 2*n - 2 items at top level and have already generated 2.
	levels[maxBits].needed = 2*n - 4

	level := maxBits
	for {
		l := &levels[level]
		if l.nextPairFreq == math.MaxInt32 && l.nextCharFreq == math.MaxInt32 {
			// We've run out of both leaves and pairs.
			// End all calculations for this level.
			// To make sure we never come back to this level or any lower level,
			// set nextPairFreq impossibly large.
			l.needed = 0
			levels[level+1].nextPairFreq = math.MaxInt32
			level++
			continue
		}

		prevFreq := l.lastFreq
		if l.nextCharFreq < l.nextPairFreq {
			// The next item on this row is a leaf node.
			n := leafCounts[level][level] + 1
			l.lastFreq = l.nextCharFreq
			// Lower leafCounts are the same of the previous node.
			leafCounts[level][level] = n
			l.nextCharFreq = list[n].freq
		} else {
			// The next item on this row is a pair from the previous row.
			// nextPairFreq isn't valid until we generate two
			// more values in the level below
			l.lastFreq = l.nextPairFreq
			// Take leaf counts from the lower level, except counts[level] remains the same.
			copy(leafCounts[level][:level], leafCounts[level-1][:level])
			levels[l.level-1].needed = 2
		}

		if l.needed--; l.needed == 0 {
			// We've done everything we need to do for this level.
			// Continue calculating one level up. Fill in nextPairFreq
			// of that level with the sum of the two nodes we've just calculated on
			// this level.
			if l.level == maxBits {
				// All done!
				break
			}
			levels[l.level+1].nextPairFreq = prevFreq + l.lastFreq
			level++
		} else {
			// If we stole from below, move down temporarily to replenish it.
			for levels[level-1].needed > 0 {
				level--
			}
		}
	}

	// Somethings is wrong if at the end, the top level is null or hasn't used
	// all of the leaves.
	if leafCounts[maxBits][maxBits] != n {
		panic("leafCounts[maxBits][maxBits] != n")
	}

	bitCount := h.bitCount[:maxBits+1]
	bits := 1
	counts := &leafCounts[maxBits]
	for level := maxBits; level > 0; level-- {
		// chain.leafCount gives the number of literals requiring at least "bits"
		// bits to encode.
		bitCount[bits] = counts[level] - counts[level-1]
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

		slices.SortFunc(chunk, func(a, b literalNode) int {
			return cmp.Compare(a.literal, b.literal)
		})
		for _, node := range chunk {
			h.codes[node.literal] = hcode{code: reverseBits(code, uint8(n)), len: uint16(n)}
			code++
		}
		list = list[0 : len(list)-int(bits)]
	}
}

// Update this Huffman Code object to be the minimum code for the specified frequency count.
//
// freq is an array of frequencies, in which freq[i] gives the frequency of literal i.
// maxBits  The maximum number of bits to use for any literal.
func (h *huffmanEncoder) generate(freq []int32, maxBits int32) {
	if h.freqcache == nil {
		// Allocate a reusable buffer with the longest possible frequency table.
		// Possible lengths are codegenCodeCount, offsetCodeCount and maxNumLit.
		// The largest of these is maxNumLit, so we allocate for that case.
		h.freqcache = make([]literalNode, maxNumLit+1)
	}
	list := h.freqcache[:len(freq)+1]
	// Number of non-zero literals
	count := 0
	// Set list to be the set of all non-zero literals and their frequencies
	for i, f := range freq {
		if f != 0 {
			list[count] = literalNode{uint16(i), f}
			count++
		} else {
			h.codes[i].len = 0
		}
	}

	list = list[:count]
	if count <= 2 {
		// Handle the small cases here, because they are awkward for the general case code. With
		// two or fewer literals, everything has bit length 1.
		for i, node := range list {
			// "list" is in order of increasing literal value.
			h.codes[node.literal].set(uint16(i), 1)
		}
		return
	}
	slices.SortFunc(list, func(a, b literalNode) int {
		if c := cmp.Compare(a.freq, b.freq); c != 0 {
			return c
		}
		return cmp.Compare(a.literal, b.literal)
	})

	// Get the number of literals for each bit count
	bitCount := h.bitCounts(list, maxBits)
	// And do the assignment
	h.assignEncodingAndSize(bitCount, list)
}

func reverseBits(number uint16, bitLength byte) uint16 {
	return bits.Reverse16(number << (16 - bitLength))
}
