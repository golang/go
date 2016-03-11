// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// Ctz64 counts trailing (low-order) zeroes,
// and if all are zero, then 64.
func Ctz64(x uint64) uint64 {
	if x&0xffffffff == 0 {
		return 32 + uint64(Ctz32(uint32(x>>32)))
	}
	return uint64(Ctz32(uint32(x)))

}

// Ctz32 counts trailing (low-order) zeroes,
// and if all are zero, then 32.
func Ctz32(x uint32) uint32 {
	if x&0xffff == 0 {
		return 16 + uint32(Ctz16(uint16(x>>16)))
	}
	return uint32(Ctz16(uint16(x)))
}

// Ctz16 counts trailing (low-order) zeroes,
// and if all are zero, then 16.
func Ctz16(x uint16) uint16 {
	if x&0xff == 0 {
		return 8 + uint16(Ctz8(uint8(x>>8)))
	}
	return uint16(Ctz8(uint8(x)))
}

// Ctz8 counts trailing (low-order) zeroes,
// and if all are zero, then 8.
func Ctz8(x uint8) uint8 {
	return ctzVals[x]
}

var ctzVals = [256]uint8{
	8, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	6, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	7, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	6, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0}

// Bswap64 returns its input with byte order reversed
// 0x0102030405060708 -> 0x0807060504030201
func Bswap64(x uint64) uint64 {
	c8 := uint64(0xff00ff00ff00ff00)
	a := (x & c8) >> 8
	b := (x &^ c8) << 8
	x = a | b
	c16 := uint64(0xffff0000ffff0000)
	a = (x & c16) >> 16
	b = (x &^ c16) << 16
	x = a | b
	c32 := uint64(0xffffffff00000000)
	a = (x & c32) >> 32
	b = (x &^ c32) << 32
	x = a | b
	return x
}

// Bswap32 returns its input with byte order reversed
// 0x01020304 -> 0x04030201
func Bswap32(x uint32) uint32 {
	c8 := uint32(0xff00ff00)
	a := (x & c8) >> 8
	b := (x &^ c8) << 8
	x = a | b
	c16 := uint32(0xffff0000)
	a = (x & c16) >> 16
	b = (x &^ c16) << 16
	x = a | b
	return x
}
