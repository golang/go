// Copyright 2024 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390xasm

import (
	"fmt"
	"strings"
)

// A BitField is a bit-field in a 64-bit double word.
// Bits are counted from 0 from the MSB to 63 as the LSB.
type BitField struct {
	Offs uint8 // the offset of the left-most bit.
	Bits uint8 // length in bits.
}

func (b BitField) String() string {
	if b.Bits > 1 {
		return fmt.Sprintf("[%d:%d]", b.Offs, int(b.Offs+b.Bits)-1)
	} else if b.Bits == 1 {
		return fmt.Sprintf("[%d]", b.Offs)
	} else {
		return fmt.Sprintf("[%d, len=0]", b.Offs)
	}
}

// Parse extracts the bitfield b from i, and return it as an unsigned integer.
// Parse will panic if b is invalid.
func (b BitField) Parse(i uint64) uint64 {
	if b.Bits > 64 || b.Bits == 0 || b.Offs > 63 || b.Offs+b.Bits > 64 {
		panic(fmt.Sprintf("invalid bitfiled %v", b))
	}
	if b.Bits == 20 {
		return ((((i >> (64 - b.Offs - b.Bits)) & ((1 << 8) - 1)) << 12) | ((i >> (64 - b.Offs - b.Bits + 8)) & 0xFFF))

	} else {
		return (i >> (64 - b.Offs - b.Bits)) & ((1 << b.Bits) - 1)
	}
}

// ParseSigned extracts the bitfield b from i, and return it as a signed integer.
// ParseSigned will panic if b is invalid.
func (b BitField) ParseSigned(i uint64) int64 {
	u := int64(b.Parse(i))
	return u << (64 - b.Bits) >> (64 - b.Bits)
}

// BitFields is a series of BitFields representing a single number.
type BitFields []BitField

func (bs BitFields) String() string {
	ss := make([]string, len(bs))
	for i, bf := range bs {
		ss[i] = bf.String()
	}
	return fmt.Sprintf("<%s>", strings.Join(ss, "|"))
}

func (bs *BitFields) Append(b BitField) {
	*bs = append(*bs, b)
}

// parse extracts the bitfields from i, concatenate them and return the result
// as an unsigned integer and the total length of all the bitfields.
// parse will panic if any bitfield in b is invalid, but it doesn't check if
// the sequence of bitfields is reasonable.
func (bs BitFields) parse(i uint64) (u uint64, Bits uint8) {
	for _, b := range bs {
		u = (u << b.Bits) | uint64(b.Parse(i))
		Bits += b.Bits
	}
	return u, Bits
}

// Parse extracts the bitfields from i, concatenate them and return the result
// as an unsigned integer. Parse will panic if any bitfield in b is invalid.
func (bs BitFields) Parse(i uint64) uint64 {
	u, _ := bs.parse(i)
	return u
}

// ParseSigned extracts the bitfields from i, concatenate them and return the result
// as a signed integer. Parse will panic if any bitfield in b is invalid.
func (bs BitFields) ParseSigned(i uint64) int64 {
	u, l := bs.parse(i)
	return int64(u) << (64 - l) >> (64 - l)
}

// Count the number of bits in the aggregate BitFields
func (bs BitFields) NumBits() int {
	num := 0
	for _, b := range bs {
		num += int(b.Bits)
	}
	return num
}
