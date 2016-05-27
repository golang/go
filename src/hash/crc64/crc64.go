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

var (
	slicing8TableISO  = makeSlicingBy8Table(makeTable(ISO))
	slicing8TableECMA = makeSlicingBy8Table(makeTable(ECMA))
)

// MakeTable returns a Table constructed from the specified polynomial.
// The contents of this Table must not be modified.
func MakeTable(poly uint64) *Table {
	switch poly {
	case ISO:
		return &slicing8TableISO[0]
	case ECMA:
		return &slicing8TableECMA[0]
	default:
		return makeTable(poly)
	}
}

func makeTable(poly uint64) *Table {
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

func makeSlicingBy8Table(t *Table) *[8]Table {
	var helperTable [8]Table
	helperTable[0] = *t
	for i := 0; i < 256; i++ {
		crc := t[i]
		for j := 1; j < 8; j++ {
			crc = t[crc&0xff] ^ (crc >> 8)
			helperTable[j][i] = crc
		}
	}
	return &helperTable
}

// digest represents the partial evaluation of a checksum.
type digest struct {
	crc uint64
	tab *Table
}

// New creates a new hash.Hash64 computing the CRC-64 checksum
// using the polynomial represented by the Table.
// Its Sum method will lay the value out in big-endian byte order.
func New(tab *Table) hash.Hash64 { return &digest{0, tab} }

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return 1 }

func (d *digest) Reset() { d.crc = 0 }

func update(crc uint64, tab *Table, p []byte) uint64 {
	crc = ^crc
	// Table comparison is somewhat expensive, so avoid it for small sizes
	for len(p) >= 64 {
		var helperTable *[8]Table
		if *tab == slicing8TableECMA[0] {
			helperTable = slicing8TableECMA
		} else if *tab == slicing8TableISO[0] {
			helperTable = slicing8TableISO
			// For smaller sizes creating extended table takes too much time
		} else if len(p) > 16384 {
			helperTable = makeSlicingBy8Table(tab)
		} else {
			break
		}
		// Update using slicing-by-8
		for len(p) > 8 {
			crc ^= uint64(p[0]) | uint64(p[1])<<8 | uint64(p[2])<<16 | uint64(p[3])<<24 |
				uint64(p[4])<<32 | uint64(p[5])<<40 | uint64(p[6])<<48 | uint64(p[7])<<56
			crc = helperTable[7][crc&0xff] ^
				helperTable[6][(crc>>8)&0xff] ^
				helperTable[5][(crc>>16)&0xff] ^
				helperTable[4][(crc>>24)&0xff] ^
				helperTable[3][(crc>>32)&0xff] ^
				helperTable[2][(crc>>40)&0xff] ^
				helperTable[1][(crc>>48)&0xff] ^
				helperTable[0][crc>>56]
			p = p[8:]
		}
	}
	// For reminders or small sizes
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
	return append(in, byte(s>>56), byte(s>>48), byte(s>>40), byte(s>>32), byte(s>>24), byte(s>>16), byte(s>>8), byte(s))
}

// Checksum returns the CRC-64 checksum of data
// using the polynomial represented by the Table.
func Checksum(data []byte, tab *Table) uint64 { return update(0, tab, data) }
