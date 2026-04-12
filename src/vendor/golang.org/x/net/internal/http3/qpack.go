// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"io"

	"golang.org/x/net/http2/hpack"
)

// QPACK (RFC 9204) header compression wire encoding.
// https://www.rfc-editor.org/rfc/rfc9204.html

// tableType is the static or dynamic table.
//
// The T bit in QPACK instructions indicates whether a table index refers to
// the dynamic (T=0) or static (T=1) table. tableTypeForTBit and tableType.tbit
// convert a T bit from the wire encoding to/from a tableType.
type tableType byte

const (
	dynamicTable = 0x00 // T=0, dynamic table
	staticTable  = 0xff // T=1, static table
)

// tableTypeForTbit returns the table type corresponding to a T bit value.
// The input parameter contains a byte masked to contain only the T bit.
func tableTypeForTbit(bit byte) tableType {
	if bit == 0 {
		return dynamicTable
	}
	return staticTable
}

// tbit produces the T bit corresponding to the table type.
// The input parameter contains a byte with the T bit set to 1,
// and the return is either the input or 0 depending on the table type.
func (t tableType) tbit(bit byte) byte {
	return bit & byte(t)
}

// indexType indicates a literal's indexing status.
//
// The N bit in QPACK instructions indicates whether a literal is "never-indexed".
// A never-indexed literal (N=1) must not be encoded as an indexed literal if it
// forwarded on another connection.
//
// (See https://www.rfc-editor.org/rfc/rfc9204.html#section-7.1 for details on the
// security reasons for never-indexed literals.)
type indexType byte

const (
	mayIndex   = 0x00 // N=0, not a never-indexed literal
	neverIndex = 0xff // N=1, never-indexed literal
)

// indexTypeForNBit returns the index type corresponding to a N bit value.
// The input parameter contains a byte masked to contain only the N bit.
func indexTypeForNBit(bit byte) indexType {
	if bit == 0 {
		return mayIndex
	}
	return neverIndex
}

// nbit produces the N bit corresponding to the table type.
// The input parameter contains a byte with the N bit set to 1,
// and the return is either the input or 0 depending on the table type.
func (t indexType) nbit(bit byte) byte {
	return bit & byte(t)
}

// Indexed Field Line:
//
//       0   1   2   3   4   5   6   7
//     +---+---+---+---+---+---+---+---+
//     | 1 | T |      Index (6+)       |
//     +---+---+-----------------------+
//
// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.5.2

func appendIndexedFieldLine(b []byte, ttype tableType, index int) []byte {
	const tbit = 0b_01000000
	return appendPrefixedInt(b, 0b_1000_0000|ttype.tbit(tbit), 6, int64(index))
}

func (st *stream) decodeIndexedFieldLine(b byte) (itype indexType, name, value string, err error) {
	index, err := st.readPrefixedIntWithByte(b, 6)
	if err != nil {
		return 0, "", "", err
	}
	const tbit = 0b_0100_0000
	if tableTypeForTbit(b&tbit) == staticTable {
		ent, err := staticTableEntry(index)
		if err != nil {
			return 0, "", "", err
		}
		return mayIndex, ent.name, ent.value, nil
	} else {
		return 0, "", "", errors.New("dynamic table is not supported yet")
	}
}

// Literal Field Line With Name Reference:
//
//      0   1   2   3   4   5   6   7
//     +---+---+---+---+---+---+---+---+
//     | 0 | 1 | N | T |Name Index (4+)|
//     +---+---+---+---+---------------+
//     | H |     Value Length (7+)     |
//     +---+---------------------------+
//     |  Value String (Length bytes)  |
//     +-------------------------------+
//
// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.5.4

func appendLiteralFieldLineWithNameReference(b []byte, ttype tableType, itype indexType, nameIndex int, value string) []byte {
	const tbit = 0b_0001_0000
	const nbit = 0b_0010_0000
	b = appendPrefixedInt(b, 0b_0100_0000|itype.nbit(nbit)|ttype.tbit(tbit), 4, int64(nameIndex))
	b = appendPrefixedString(b, 0, 7, value)
	return b
}

func (st *stream) decodeLiteralFieldLineWithNameReference(b byte) (itype indexType, name, value string, err error) {
	nameIndex, err := st.readPrefixedIntWithByte(b, 4)
	if err != nil {
		return 0, "", "", err
	}

	const tbit = 0b_0001_0000
	if tableTypeForTbit(b&tbit) == staticTable {
		ent, err := staticTableEntry(nameIndex)
		if err != nil {
			return 0, "", "", err
		}
		name = ent.name
	} else {
		return 0, "", "", errors.New("dynamic table is not supported yet")
	}

	_, value, err = st.readPrefixedString(7)
	if err != nil {
		return 0, "", "", err
	}

	const nbit = 0b_0010_0000
	itype = indexTypeForNBit(b & nbit)

	return itype, name, value, nil
}

// Literal Field Line with Literal Name:
//
//       0   1   2   3   4   5   6   7
//     +---+---+---+---+---+---+---+---+
//     | 0 | 0 | 1 | N | H |NameLen(3+)|
//     +---+---+---+---+---+-----------+
//     |  Name String (Length bytes)   |
//     +---+---------------------------+
//     | H |     Value Length (7+)     |
//     +---+---------------------------+
//     |  Value String (Length bytes)  |
//     +-------------------------------+
//
// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.5.6

func appendLiteralFieldLineWithLiteralName(b []byte, itype indexType, name, value string) []byte {
	const nbit = 0b_0001_0000
	b = appendPrefixedString(b, 0b_0010_0000|itype.nbit(nbit), 3, name)
	b = appendPrefixedString(b, 0, 7, value)
	return b
}

func (st *stream) decodeLiteralFieldLineWithLiteralName(b byte) (itype indexType, name, value string, err error) {
	name, err = st.readPrefixedStringWithByte(b, 3)
	if err != nil {
		return 0, "", "", err
	}
	_, value, err = st.readPrefixedString(7)
	if err != nil {
		return 0, "", "", err
	}
	const nbit = 0b_0001_0000
	itype = indexTypeForNBit(b & nbit)
	return itype, name, value, nil
}

// Prefixed-integer encoding from RFC 7541, section 5.1
//
// Prefixed integers consist of some number of bits of data,
// N bits of encoded integer, and 0 or more additional bytes of
// encoded integer.
//
// The RFCs represent this as, for example:
//
//       0   1   2   3   4   5   6   7
//     +---+---+---+---+---+---+---+---+
//     | 0 | 0 | 1 |   Capacity (5+)   |
//     +---+---+---+-------------------+
//
// "Capacity" is an integer with a 5-bit prefix.
//
// In the following functions, a "prefixLen" parameter is the number
// of integer bits in the first byte (5 in the above example), and
// a "firstByte" parameter is a byte containing the first byte of
// the encoded value (0x001x_xxxx in the above example).
//
// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.1.1
// https://www.rfc-editor.org/rfc/rfc7541#section-5.1

// readPrefixedInt reads an RFC 7541 prefixed integer from st.
func (st *stream) readPrefixedInt(prefixLen uint8) (firstByte byte, v int64, err error) {
	firstByte, err = st.ReadByte()
	if err != nil {
		return 0, 0, errQPACKDecompressionFailed
	}
	v, err = st.readPrefixedIntWithByte(firstByte, prefixLen)
	return firstByte, v, err
}

// readPrefixedIntWithByte reads an RFC 7541 prefixed integer from st.
// The first byte has already been read from the stream.
func (st *stream) readPrefixedIntWithByte(firstByte byte, prefixLen uint8) (v int64, err error) {
	prefixMask := (byte(1) << prefixLen) - 1
	v = int64(firstByte & prefixMask)
	if v != int64(prefixMask) {
		return v, nil
	}
	m := 0
	for {
		b, err := st.ReadByte()
		if err != nil {
			return 0, errQPACKDecompressionFailed
		}
		v += int64(b&127) << m
		m += 7
		if b&128 == 0 {
			break
		}
	}
	return v, err
}

// appendPrefixedInt appends an RFC 7541 prefixed integer to b.
//
// The firstByte parameter includes the non-integer bits of the first byte.
// The other bits must be zero.
func appendPrefixedInt(b []byte, firstByte byte, prefixLen uint8, i int64) []byte {
	u := uint64(i)
	prefixMask := (uint64(1) << prefixLen) - 1
	if u < prefixMask {
		return append(b, firstByte|byte(u))
	}
	b = append(b, firstByte|byte(prefixMask))
	u -= prefixMask
	for u >= 128 {
		b = append(b, 0x80|byte(u&0x7f))
		u >>= 7
	}
	return append(b, byte(u))
}

// String literal encoding from RFC 7541, section 5.2
//
// String literals consist of a single bit flag indicating
// whether the string is Huffman-encoded, a prefixed integer (see above),
// and the string.
//
// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.1.2
// https://www.rfc-editor.org/rfc/rfc7541#section-5.2

// readPrefixedString reads an RFC 7541 string from st.
func (st *stream) readPrefixedString(prefixLen uint8) (firstByte byte, s string, err error) {
	firstByte, err = st.ReadByte()
	if err != nil {
		return 0, "", errQPACKDecompressionFailed
	}
	s, err = st.readPrefixedStringWithByte(firstByte, prefixLen)
	return firstByte, s, err
}

// readPrefixedStringWithByte reads an RFC 7541 string from st.
// The first byte has already been read from the stream.
func (st *stream) readPrefixedStringWithByte(firstByte byte, prefixLen uint8) (s string, err error) {
	size, err := st.readPrefixedIntWithByte(firstByte, prefixLen)
	if err != nil {
		return "", errQPACKDecompressionFailed
	}

	hbit := byte(1) << prefixLen
	isHuffman := firstByte&hbit != 0

	// TODO: Avoid allocating here.
	data := make([]byte, size)
	if _, err := io.ReadFull(st, data); err != nil {
		return "", errQPACKDecompressionFailed
	}
	if isHuffman {
		// TODO: Move Huffman functions into a new package that hpack (HTTP/2)
		// and this package can both import. Most of the hpack package isn't
		// relevant to HTTP/3.
		s, err := hpack.HuffmanDecodeToString(data)
		if err != nil {
			return "", errQPACKDecompressionFailed
		}
		return s, nil
	}
	return string(data), nil
}

// appendPrefixedString appends an RFC 7541 string to st,
// applying Huffman encoding and setting the H bit (indicating Huffman encoding)
// when appropriate.
//
// The firstByte parameter includes the non-integer bits of the first byte.
// The other bits must be zero.
func appendPrefixedString(b []byte, firstByte byte, prefixLen uint8, s string) []byte {
	huffmanLen := hpack.HuffmanEncodeLength(s)
	if huffmanLen < uint64(len(s)) {
		hbit := byte(1) << prefixLen
		b = appendPrefixedInt(b, firstByte|hbit, prefixLen, int64(huffmanLen))
		b = hpack.AppendHuffmanString(b, s)
	} else {
		b = appendPrefixedInt(b, firstByte, prefixLen, int64(len(s)))
		b = append(b, s...)
	}
	return b
}
