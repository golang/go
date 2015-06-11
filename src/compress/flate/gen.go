// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program generates fixedhuff.go
// Invoke as
//
//	go run gen.go -output fixedhuff.go

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
)

var filename = flag.String("output", "fixedhuff.go", "output file name")

const maxCodeLen = 16

// Note: the definition of the huffmanDecoder struct is copied from
// inflate.go, as it is private to the implementation.

// chunk & 15 is number of bits
// chunk >> 4 is value, including table link

const (
	huffmanChunkBits  = 9
	huffmanNumChunks  = 1 << huffmanChunkBits
	huffmanCountMask  = 15
	huffmanValueShift = 4
)

type huffmanDecoder struct {
	min      int                      // the minimum code length
	chunks   [huffmanNumChunks]uint32 // chunks as described above
	links    [][]uint32               // overflow links
	linkMask uint32                   // mask the width of the link table
}

// Initialize Huffman decoding tables from array of code lengths.
// Following this function, h is guaranteed to be initialized into a complete
// tree (i.e., neither over-subscribed nor under-subscribed). The exception is a
// degenerate case where the tree has only a single symbol with length 1. Empty
// trees are permitted.
func (h *huffmanDecoder) init(bits []int) bool {
	// Sanity enables additional runtime tests during Huffman
	// table construction.  It's intended to be used during
	// development to supplement the currently ad-hoc unit tests.
	const sanity = false

	if h.min != 0 {
		*h = huffmanDecoder{}
	}

	// Count number of codes of each length,
	// compute min and max length.
	var count [maxCodeLen]int
	var min, max int
	for _, n := range bits {
		if n == 0 {
			continue
		}
		if min == 0 || n < min {
			min = n
		}
		if n > max {
			max = n
		}
		count[n]++
	}

	// Empty tree. The decompressor.huffSym function will fail later if the tree
	// is used. Technically, an empty tree is only valid for the HDIST tree and
	// not the HCLEN and HLIT tree. However, a stream with an empty HCLEN tree
	// is guaranteed to fail since it will attempt to use the tree to decode the
	// codes for the HLIT and HDIST trees. Similarly, an empty HLIT tree is
	// guaranteed to fail later since the compressed data section must be
	// composed of at least one symbol (the end-of-block marker).
	if max == 0 {
		return true
	}

	code := 0
	var nextcode [maxCodeLen]int
	for i := min; i <= max; i++ {
		code <<= 1
		nextcode[i] = code
		code += count[i]
	}

	// Check that the coding is complete (i.e., that we've
	// assigned all 2-to-the-max possible bit sequences).
	// Exception: To be compatible with zlib, we also need to
	// accept degenerate single-code codings.  See also
	// TestDegenerateHuffmanCoding.
	if code != 1<<uint(max) && !(code == 1 && max == 1) {
		return false
	}

	h.min = min
	if max > huffmanChunkBits {
		numLinks := 1 << (uint(max) - huffmanChunkBits)
		h.linkMask = uint32(numLinks - 1)

		// create link tables
		link := nextcode[huffmanChunkBits+1] >> 1
		h.links = make([][]uint32, huffmanNumChunks-link)
		for j := uint(link); j < huffmanNumChunks; j++ {
			reverse := int(reverseByte[j>>8]) | int(reverseByte[j&0xff])<<8
			reverse >>= uint(16 - huffmanChunkBits)
			off := j - uint(link)
			if sanity && h.chunks[reverse] != 0 {
				panic("impossible: overwriting existing chunk")
			}
			h.chunks[reverse] = uint32(off<<huffmanValueShift | (huffmanChunkBits + 1))
			h.links[off] = make([]uint32, numLinks)
		}
	}

	for i, n := range bits {
		if n == 0 {
			continue
		}
		code := nextcode[n]
		nextcode[n]++
		chunk := uint32(i<<huffmanValueShift | n)
		reverse := int(reverseByte[code>>8]) | int(reverseByte[code&0xff])<<8
		reverse >>= uint(16 - n)
		if n <= huffmanChunkBits {
			for off := reverse; off < len(h.chunks); off += 1 << uint(n) {
				// We should never need to overwrite
				// an existing chunk.  Also, 0 is
				// never a valid chunk, because the
				// lower 4 "count" bits should be
				// between 1 and 15.
				if sanity && h.chunks[off] != 0 {
					panic("impossible: overwriting existing chunk")
				}
				h.chunks[off] = chunk
			}
		} else {
			j := reverse & (huffmanNumChunks - 1)
			if sanity && h.chunks[j]&huffmanCountMask != huffmanChunkBits+1 {
				// Longer codes should have been
				// associated with a link table above.
				panic("impossible: not an indirect chunk")
			}
			value := h.chunks[j] >> huffmanValueShift
			linktab := h.links[value]
			reverse >>= huffmanChunkBits
			for off := reverse; off < len(linktab); off += 1 << uint(n-huffmanChunkBits) {
				if sanity && linktab[off] != 0 {
					panic("impossible: overwriting existing chunk")
				}
				linktab[off] = chunk
			}
		}
	}

	if sanity {
		// Above we've sanity checked that we never overwrote
		// an existing entry.  Here we additionally check that
		// we filled the tables completely.
		for i, chunk := range h.chunks {
			if chunk == 0 {
				// As an exception, in the degenerate
				// single-code case, we allow odd
				// chunks to be missing.
				if code == 1 && i%2 == 1 {
					continue
				}
				panic("impossible: missing chunk")
			}
		}
		for _, linktab := range h.links {
			for _, chunk := range linktab {
				if chunk == 0 {
					panic("impossible: missing chunk")
				}
			}
		}
	}

	return true
}

func main() {
	flag.Parse()

	var h huffmanDecoder
	var bits [288]int
	initReverseByte()
	for i := 0; i < 144; i++ {
		bits[i] = 8
	}
	for i := 144; i < 256; i++ {
		bits[i] = 9
	}
	for i := 256; i < 280; i++ {
		bits[i] = 7
	}
	for i := 280; i < 288; i++ {
		bits[i] = 8
	}
	h.init(bits[:])
	if h.links != nil {
		log.Fatal("Unexpected links table in fixed Huffman decoder")
	}

	var buf bytes.Buffer

	fmt.Fprintf(&buf, `// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.`+"\n\n")

	fmt.Fprintln(&buf, "package flate")
	fmt.Fprintln(&buf)
	fmt.Fprintln(&buf, "// autogenerated by go run gen.go -output fixedhuff.go, DO NOT EDIT")
	fmt.Fprintln(&buf)
	fmt.Fprintln(&buf, "var fixedHuffmanDecoder = huffmanDecoder{")
	fmt.Fprintf(&buf, "\t%d,\n", h.min)
	fmt.Fprintln(&buf, "\t[huffmanNumChunks]uint32{")
	for i := 0; i < huffmanNumChunks; i++ {
		if i&7 == 0 {
			fmt.Fprintf(&buf, "\t\t")
		} else {
			fmt.Fprintf(&buf, " ")
		}
		fmt.Fprintf(&buf, "0x%04x,", h.chunks[i])
		if i&7 == 7 {
			fmt.Fprintln(&buf)
		}
	}
	fmt.Fprintln(&buf, "\t},")
	fmt.Fprintln(&buf, "\tnil, 0,")
	fmt.Fprintln(&buf, "}")

	data, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	err = ioutil.WriteFile(*filename, data, 0644)
	if err != nil {
		log.Fatal(err)
	}
}

var reverseByte [256]byte

func initReverseByte() {
	for x := 0; x < 256; x++ {
		var result byte
		for i := uint(0); i < 8; i++ {
			result |= byte(((x >> i) & 1) << (7 - i))
		}
		reverseByte[x] = result
	}
}
