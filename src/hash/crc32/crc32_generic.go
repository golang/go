// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains CRC32 algorithms that are not specific to any architecture
// and don't use hardware acceleration.
//
// The simple (and slow) CRC32 implementation only uses a 256*4 bytes table.
//
// The slicing-by-8 algorithm is a faster implementation that uses a bigger
// table (8*256*4 bytes).

package crc32

// simpleMakeTable allocates and constructs a Table for the specified
// polynomial. The table is suitable for use with the simple algorithm
// (simpleUpdate).
func simpleMakeTable(poly uint32) *Table {
	t := new(Table)
	simplePopulateTable(poly, t)
	return t
}

// simplePopulateTable constructs a Table for the specified polynomial, suitable
// for use with simpleUpdate.
func simplePopulateTable(poly uint32, t *Table) {
	for i := 0; i < 256; i++ {
		crc := uint32(i)
		for j := 0; j < 8; j++ {
			if crc&1 == 1 {
				crc = (crc >> 1) ^ poly
			} else {
				crc >>= 1
			}
		}
		t[i] = crc
	}
}

// simpleUpdate uses the simple algorithm to update the CRC, given a table that
// was previously computed using simpleMakeTable.
func simpleUpdate(crc uint32, tab *Table, p []byte) uint32 {
	crc = ^crc
	for _, v := range p {
		crc = tab[byte(crc)^v] ^ (crc >> 8)
	}
	return ^crc
}

// Use slicing-by-8 when payload >= this value.
const slicing8Cutoff = 16

// slicing8Table is array of 8 Tables, used by the slicing-by-8 algorithm.
type slicing8Table [8]Table

// slicingMakeTable constructs a slicing8Table for the specified polynomial. The
// table is suitable for use with the slicing-by-8 algorithm (slicingUpdate).
func slicingMakeTable(poly uint32) *slicing8Table {
	t := new(slicing8Table)
	simplePopulateTable(poly, &t[0])
	for i := 0; i < 256; i++ {
		crc := t[0][i]
		for j := 1; j < 8; j++ {
			crc = t[0][crc&0xFF] ^ (crc >> 8)
			t[j][i] = crc
		}
	}
	return t
}

// slicingUpdate uses the slicing-by-8 algorithm to update the CRC, given a
// table that was previously computed using slicingMakeTable.
func slicingUpdate(crc uint32, tab *slicing8Table, p []byte) uint32 {
	if len(p) >= slicing8Cutoff {
		crc = ^crc
		for len(p) > 8 {
			crc ^= uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
			crc = tab[0][p[7]] ^ tab[1][p[6]] ^ tab[2][p[5]] ^ tab[3][p[4]] ^
				tab[4][crc>>24] ^ tab[5][(crc>>16)&0xFF] ^
				tab[6][(crc>>8)&0xFF] ^ tab[7][crc&0xFF]
			p = p[8:]
		}
		crc = ^crc
	}
	if len(p) == 0 {
		return crc
	}
	return simpleUpdate(crc, &tab[0], p)
}
