// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crc64 implements the 64-bit cyclic redundancy check, or CRC-64,
// checksum. See http://en.wikipedia.org/wiki/Cyclic_redundancy_check for
// information.
package crc64

import "hash"

// The size of a CRC-64 checksum in bytes.
const Size = 8

// Predefined polynomials.
const (
	// The ISO polynomial, defined in ISO 3309 and used in HDLC.
	ISO = 0xD800000000000000

	// The ECMA polynomial, defined in ECMA 182.
	ECMA = 0xC96C5795D7870F42
)

// Table is a 256-word table representing the polynomial for efficient processing.
type Table [256]uint64

// MakeTable returns the Table constructed from the specified polynomial.
func MakeTable(poly uint64) *Table {
	t := new(Table)
	for i := 0; i < 256; i++ {
		crc := uint64(i)
		for j := 0; j < 8; j++ {
			if crc&1 == 1 {
				crc = (crc >> 1) ^ poly
			} else {
				crc >>= 1
			}
		}
		t[i] = crc
	}
	return t
}

// digest represents the partial evaluation of a checksum.
type digest struct {
	crc uint64
	tab *Table
}

// New creates a new hash.Hash64 computing the CRC-64 checksum
// using the polynomial represented by the Table.
func New(tab *Table) hash.Hash64 { return &digest{0, tab} }

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return 1 }

func (d *digest) Reset() { d.crc = 0 }

func update(crc uint64, tab *Table, p []byte) uint64 {
	crc = ^crc
	for _, v := range p {
		crc = tab[byte(crc)^v] ^ (crc >> 8)
	}
	return ^crc
}

// Update returns the result of adding the bytes in p to the crc.
func Update(crc uint64, tab *Table, p []byte) uint64 {
	return update(crc, tab, p)
}

func (d *digest) Write(p []byte) (n int, err error) {
	d.crc = update(d.crc, d.tab, p)
	return len(p), nil
}

func (d *digest) Sum64() uint64 { return d.crc }

func (d *digest) Sum(in []byte) []byte {
	s := d.Sum64()
	in = append(in, byte(s>>56))
	in = append(in, byte(s>>48))
	in = append(in, byte(s>>40))
	in = append(in, byte(s>>32))
	in = append(in, byte(s>>24))
	in = append(in, byte(s>>16))
	in = append(in, byte(s>>8))
	in = append(in, byte(s))
	return in
}

// Checksum returns the CRC-64 checksum of data
// using the polynomial represented by the Table.
func Checksum(data []byte, tab *Table) uint64 { return update(0, tab, data) }
