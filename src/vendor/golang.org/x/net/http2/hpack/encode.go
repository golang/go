// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpack

import (
	"io"
)

const (
	uint32Max              = ^uint32(0)
	initialHeaderTableSize = 4096
)

type Encoder struct {
	dynTab dynamicTable
	// minSize is the minimum table size set by
	// SetMaxDynamicTableSize after the previous Header Table Size
	// Update.
	minSize uint32
	// maxSizeLimit is the maximum table size this encoder
	// supports. This will protect the encoder from too large
	// size.
	maxSizeLimit uint32
	// tableSizeUpdate indicates whether "Header Table Size
	// Update" is required.
	tableSizeUpdate bool
	w               io.Writer
	buf             []byte
}

// NewEncoder returns a new Encoder which performs HPACK encoding. An
// encoded data is written to w.
func NewEncoder(w io.Writer) *Encoder {
	e := &Encoder{
		minSize:         uint32Max,
		maxSizeLimit:    initialHeaderTableSize,
		tableSizeUpdate: false,
		w:               w,
	}
	e.dynTab.setMaxSize(initialHeaderTableSize)
	return e
}

// WriteField encodes f into a single Write to e's underlying Writer.
// This function may also produce bytes for "Header Table Size Update"
// if necessary.  If produced, it is done before encoding f.
func (e *Encoder) WriteField(f HeaderField) error {
	e.buf = e.buf[:0]

	if e.tableSizeUpdate {
		e.tableSizeUpdate = false
		if e.minSize < e.dynTab.maxSize {
			e.buf = appendTableSize(e.buf, e.minSize)
		}
		e.minSize = uint32Max
		e.buf = appendTableSize(e.buf, e.dynTab.maxSize)
	}

	idx, nameValueMatch := e.searchTable(f)
	if nameValueMatch {
		e.buf = appendIndexed(e.buf, idx)
	} else {
		indexing := e.shouldIndex(f)
		if indexing {
			e.dynTab.add(f)
		}

		if idx == 0 {
			e.buf = appendNewName(e.buf, f, indexing)
		} else {
			e.buf = appendIndexedName(e.buf, f, idx, indexing)
		}
	}
	n, err := e.w.Write(e.buf)
	if err == nil && n != len(e.buf) {
		err = io.ErrShortWrite
	}
	return err
}

// searchTable searches f in both stable and dynamic header tables.
// The static header table is searched first. Only when there is no
// exact match for both name and value, the dynamic header table is
// then searched. If there is no match, i is 0. If both name and value
// match, i is the matched index and nameValueMatch becomes true. If
// only name matches, i points to that index and nameValueMatch
// becomes false.
func (e *Encoder) searchTable(f HeaderField) (i uint64, nameValueMatch bool) {
	for idx, hf := range staticTable {
		if !constantTimeStringCompare(hf.Name, f.Name) {
			continue
		}
		if i == 0 {
			i = uint64(idx + 1)
		}
		if f.Sensitive {
			continue
		}
		if !constantTimeStringCompare(hf.Value, f.Value) {
			continue
		}
		i = uint64(idx + 1)
		nameValueMatch = true
		return
	}

	j, nameValueMatch := e.dynTab.search(f)
	if nameValueMatch || (i == 0 && j != 0) {
		i = j + uint64(len(staticTable))
	}
	return
}

// SetMaxDynamicTableSize changes the dynamic header table size to v.
// The actual size is bounded by the value passed to
// SetMaxDynamicTableSizeLimit.
func (e *Encoder) SetMaxDynamicTableSize(v uint32) {
	if v > e.maxSizeLimit {
		v = e.maxSizeLimit
	}
	if v < e.minSize {
		e.minSize = v
	}
	e.tableSizeUpdate = true
	e.dynTab.setMaxSize(v)
}

// SetMaxDynamicTableSizeLimit changes the maximum value that can be
// specified in SetMaxDynamicTableSize to v. By default, it is set to
// 4096, which is the same size of the default dynamic header table
// size described in HPACK specification. If the current maximum
// dynamic header table size is strictly greater than v, "Header Table
// Size Update" will be done in the next WriteField call and the
// maximum dynamic header table size is truncated to v.
func (e *Encoder) SetMaxDynamicTableSizeLimit(v uint32) {
	e.maxSizeLimit = v
	if e.dynTab.maxSize > v {
		e.tableSizeUpdate = true
		e.dynTab.setMaxSize(v)
	}
}

// shouldIndex reports whether f should be indexed.
func (e *Encoder) shouldIndex(f HeaderField) bool {
	return !f.Sensitive && f.Size() <= e.dynTab.maxSize
}

// appendIndexed appends index i, as encoded in "Indexed Header Field"
// representation, to dst and returns the extended buffer.
func appendIndexed(dst []byte, i uint64) []byte {
	first := len(dst)
	dst = appendVarInt(dst, 7, i)
	dst[first] |= 0x80
	return dst
}

// appendNewName appends f, as encoded in one of "Literal Header field
// - New Name" representation variants, to dst and returns the
// extended buffer.
//
// If f.Sensitive is true, "Never Indexed" representation is used. If
// f.Sensitive is false and indexing is true, "Inremental Indexing"
// representation is used.
func appendNewName(dst []byte, f HeaderField, indexing bool) []byte {
	dst = append(dst, encodeTypeByte(indexing, f.Sensitive))
	dst = appendHpackString(dst, f.Name)
	return appendHpackString(dst, f.Value)
}

// appendIndexedName appends f and index i referring indexed name
// entry, as encoded in one of "Literal Header field - Indexed Name"
// representation variants, to dst and returns the extended buffer.
//
// If f.Sensitive is true, "Never Indexed" representation is used. If
// f.Sensitive is false and indexing is true, "Incremental Indexing"
// representation is used.
func appendIndexedName(dst []byte, f HeaderField, i uint64, indexing bool) []byte {
	first := len(dst)
	var n byte
	if indexing {
		n = 6
	} else {
		n = 4
	}
	dst = appendVarInt(dst, n, i)
	dst[first] |= encodeTypeByte(indexing, f.Sensitive)
	return appendHpackString(dst, f.Value)
}

// appendTableSize appends v, as encoded in "Header Table Size Update"
// representation, to dst and returns the extended buffer.
func appendTableSize(dst []byte, v uint32) []byte {
	first := len(dst)
	dst = appendVarInt(dst, 5, uint64(v))
	dst[first] |= 0x20
	return dst
}

// appendVarInt appends i, as encoded in variable integer form using n
// bit prefix, to dst and returns the extended buffer.
//
// See
// http://http2.github.io/http2-spec/compression.html#integer.representation
func appendVarInt(dst []byte, n byte, i uint64) []byte {
	k := uint64((1 << n) - 1)
	if i < k {
		return append(dst, byte(i))
	}
	dst = append(dst, byte(k))
	i -= k
	for ; i >= 128; i >>= 7 {
		dst = append(dst, byte(0x80|(i&0x7f)))
	}
	return append(dst, byte(i))
}

// appendHpackString appends s, as encoded in "String Literal"
// representation, to dst and returns the the extended buffer.
//
// s will be encoded in Huffman codes only when it produces strictly
// shorter byte string.
func appendHpackString(dst []byte, s string) []byte {
	huffmanLength := HuffmanEncodeLength(s)
	if huffmanLength < uint64(len(s)) {
		first := len(dst)
		dst = appendVarInt(dst, 7, huffmanLength)
		dst = AppendHuffmanString(dst, s)
		dst[first] |= 0x80
	} else {
		dst = appendVarInt(dst, 7, uint64(len(s)))
		dst = append(dst, s...)
	}
	return dst
}

// encodeTypeByte returns type byte. If sensitive is true, type byte
// for "Never Indexed" representation is returned. If sensitive is
// false and indexing is true, type byte for "Incremental Indexing"
// representation is returned. Otherwise, type byte for "Without
// Indexing" is returned.
func encodeTypeByte(indexing, sensitive bool) byte {
	if sensitive {
		return 0x10
	}
	if indexing {
		return 0x40
	}
	return 0
}
