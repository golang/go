// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"fmt"
	"strings"
)

// A BitField is a bit-field in a 32-bit word.
// Bits are counted from 0 from the MSB to 31 as the LSB.
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
func (b BitField) Parse(i uint32) uint32 {
	if b.Bits > 32 || b.Bits == 0 || b.Offs > 31 || b.Offs+b.Bits > 32 {
		panic(fmt.Sprintf("invalid bitfiled %v", b))
	}
	return (i >> (32 - b.Offs - b.Bits)) & ((1 << b.Bits) - 1)
}

// ParseSigned extracts the bitfield b from i, and return it as a signed integer.
// ParseSigned will panic if b is invalid.
func (b BitField) ParseSigned(i uint32) int32 {
	u := int32(b.Parse(i))
	return u << (32 - b.Bits) >> (32 - b.Bits)
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
func (bs BitFields) parse(i uint32) (u uint32, Bits uint8) {
	for _, b := range bs {
		u = (u << b.Bits) | b.Parse(i)
		Bits += b.Bits
	}
	return u, Bits
}

// Parse extracts the bitfields from i, concatenate them and return the result
// as an unsigned integer. Parse will panic if any bitfield in b is invalid.
func (bs BitFields) Parse(i uint32) uint32 {
	u, _ := bs.parse(i)
	return u
}

// Parse extracts the bitfields from i, concatenate them and return the result
// as a signed integer. Parse will panic if any bitfield in b is invalid.
func (bs BitFields) ParseSigned(i uint32) int32 {
	u, l := bs.parse(i)
	return int32(u) << (32 - l) >> (32 - l)
}
