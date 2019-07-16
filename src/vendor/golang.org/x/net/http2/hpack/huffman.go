// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpack

import (
	"bytes"
	"errors"
	"io"
	"sync"
)

var bufPool = sync.Pool{
	New: func() interface{} { return new(bytes.Buffer) },
}

// HuffmanDecode decodes the string in v and writes the expanded
// result to w, returning the number of bytes written to w and the
// Write call's return value. At most one Write call is made.
func HuffmanDecode(w io.Writer, v []byte) (int, error) {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)
	if err := huffmanDecode(buf, 0, v); err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

// HuffmanDecodeToString decodes the string in v.
func HuffmanDecodeToString(v []byte) (string, error) {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)
	if err := huffmanDecode(buf, 0, v); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// ErrInvalidHuffman is returned for errors found decoding
// Huffman-encoded strings.
var ErrInvalidHuffman = errors.New("hpack: invalid Huffman-encoded data")

// huffmanDecode decodes v to buf.
// If maxLen is greater than 0, attempts to write more to buf than
// maxLen bytes will return ErrStringLength.
func huffmanDecode(buf *bytes.Buffer, maxLen int, v []byte) error {
	rootHuffmanNode := getRootHuffmanNode()
	n := rootHuffmanNode
	// cur is the bit buffer that has not been fed into n.
	// cbits is the number of low order bits in cur that are valid.
	// sbits is the number of bits of the symbol prefix being decoded.
	cur, cbits, sbits := uint(0), uint8(0), uint8(0)
	for _, b := range v {
		cur = cur<<8 | uint(b)
		cbits += 8
		sbits += 8
		for cbits >= 8 {
			idx := byte(cur >> (cbits - 8))
			n = n.children[idx]
			if n == nil {
				return ErrInvalidHuffman
			}
			if n.children == nil {
				if maxLen != 0 && buf.Len() == maxLen {
					return ErrStringLength
				}
				buf.WriteByte(n.sym)
				cbits -= n.codeLen
				n = rootHuffmanNode
				sbits = cbits
			} else {
				cbits -= 8
			}
		}
	}
	for cbits > 0 {
		n = n.children[byte(cur<<(8-cbits))]
		if n == nil {
			return ErrInvalidHuffman
		}
		if n.children != nil || n.codeLen > cbits {
			break
		}
		if maxLen != 0 && buf.Len() == maxLen {
			return ErrStringLength
		}
		buf.WriteByte(n.sym)
		cbits -= n.codeLen
		n = rootHuffmanNode
		sbits = cbits
	}
	if sbits > 7 {
		// Either there was an incomplete symbol, or overlong padding.
		// Both are decoding errors per RFC 7541 section 5.2.
		return ErrInvalidHuffman
	}
	if mask := uint(1<<cbits - 1); cur&mask != mask {
		// Trailing bits must be a prefix of EOS per RFC 7541 section 5.2.
		return ErrInvalidHuffman
	}

	return nil
}

type node struct {
	// children is non-nil for internal nodes
	children *[256]*node

	// The following are only valid if children is nil:
	codeLen uint8 // number of bits that led to the output of sym
	sym     byte  // output symbol
}

func newInternalNode() *node {
	return &node{children: new([256]*node)}
}

var (
	buildRootOnce       sync.Once
	lazyRootHuffmanNode *node
)

func getRootHuffmanNode() *node {
	buildRootOnce.Do(buildRootHuffmanNode)
	return lazyRootHuffmanNode
}

func buildRootHuffmanNode() {
	if len(huffmanCodes) != 256 {
		panic("unexpected size")
	}
	lazyRootHuffmanNode = newInternalNode()
	for i, code := range huffmanCodes {
		addDecoderNode(byte(i), code, huffmanCodeLen[i])
	}
}

func addDecoderNode(sym byte, code uint32, codeLen uint8) {
	cur := lazyRootHuffmanNode
	for codeLen > 8 {
		codeLen -= 8
		i := uint8(code >> codeLen)
		if cur.children[i] == nil {
			cur.children[i] = newInternalNode()
		}
		cur = cur.children[i]
	}
	shift := 8 - codeLen
	start, end := int(uint8(code<<shift)), int(1<<shift)
	for i := start; i < start+end; i++ {
		cur.children[i] = &node{sym: sym, codeLen: codeLen}
	}
}

// AppendHuffmanString appends s, as encoded in Huffman codes, to dst
// and returns the extended buffer.
func AppendHuffmanString(dst []byte, s string) []byte {
	rembits := uint8(8)

	for i := 0; i < len(s); i++ {
		if rembits == 8 {
			dst = append(dst, 0)
		}
		dst, rembits = appendByteToHuffmanCode(dst, rembits, s[i])
	}

	if rembits < 8 {
		// special EOS symbol
		code := uint32(0x3fffffff)
		nbits := uint8(30)

		t := uint8(code >> (nbits - rembits))
		dst[len(dst)-1] |= t
	}

	return dst
}

// HuffmanEncodeLength returns the number of bytes required to encode
// s in Huffman codes. The result is round up to byte boundary.
func HuffmanEncodeLength(s string) uint64 {
	n := uint64(0)
	for i := 0; i < len(s); i++ {
		n += uint64(huffmanCodeLen[s[i]])
	}
	return (n + 7) / 8
}

// appendByteToHuffmanCode appends Huffman code for c to dst and
// returns the extended buffer and the remaining bits in the last
// element. The appending is not byte aligned and the remaining bits
// in the last element of dst is given in rembits.
func appendByteToHuffmanCode(dst []byte, rembits uint8, c byte) ([]byte, uint8) {
	code := huffmanCodes[c]
	nbits := huffmanCodeLen[c]

	for {
		if rembits > nbits {
			t := uint8(code << (rembits - nbits))
			dst[len(dst)-1] |= t
			rembits -= nbits
			break
		}

		t := uint8(code >> (nbits - rembits))
		dst[len(dst)-1] |= t

		nbits -= rembits
		rembits = 8

		if nbits == 0 {
			break
		}

		dst = append(dst, 0)
	}

	return dst, rembits
}
