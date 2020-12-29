// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// crc64 包实现了64 位循环冗余校验或者 CRC-64 校验和算法。 参见：https://en.wikipedia.org/wiki/Cyclic_redundancy_check。
package crc64

import (
	"errors"
	"hash"
	"sync"
)

// CRC-64 校验和的字节长度
const Size = 8

// 预定义的多项式
const (
	// ISO 多项式，在 ISO 3309 中定义并在 HDLC 中使用。
	ISO = 0xD800000000000000

	// ECMA 多项式，在 ECMA 182 中定义。
	ECMA = 0xC96C5795D7870F42
)

// Table 是一个长度为 256 的 uint64 切片，代表一个高效运作的多项式。
type Table [256]uint64

var (
	slicing8TablesBuildOnce sync.Once
	slicing8TableISO        *[8]Table
	slicing8TableECMA       *[8]Table
)

func buildSlicing8TablesOnce() {
	slicing8TablesBuildOnce.Do(buildSlicing8Tables)
}

func buildSlicing8Tables() {
	slicing8TableISO = makeSlicingBy8Table(makeTable(ISO))
	slicing8TableECMA = makeSlicingBy8Table(makeTable(ECMA))
}

// MakeTable 返回根据指定多项式构造的 table。 该表的内容不得修改
func MakeTable(poly uint64) *Table {
	buildSlicing8TablesOnce()
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

// New 创建一个 使用 table 表示的多项式计算 CRC-64 校验和的 hash.Hash64。
// 它的 Sum 方法将按照 big-endian 字节顺序排列值。返回的 Hash64 也实现了
// encoding.BinaryMarshaler 和 encoding.BinaryUnmarshaler 来封装和取消封装hash的内部状态。
func New(tab *Table) hash.Hash64 { return &digest{0, tab} }

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return 1 }

func (d *digest) Reset() { d.crc = 0 }

const (
	magic         = "crc\x02"
	marshaledSize = len(magic) + 8 + 8
)

func (d *digest) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, marshaledSize)
	b = append(b, magic...)
	b = appendUint64(b, tableSum(d.tab))
	b = appendUint64(b, d.crc)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic) || string(b[:len(magic)]) != magic {
		return errors.New("hash/crc64: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("hash/crc64: invalid hash state size")
	}
	if tableSum(d.tab) != readUint64(b[4:]) {
		return errors.New("hash/crc64: tables do not match")
	}
	d.crc = readUint64(b[12:])
	return nil
}

func appendUint64(b []byte, x uint64) []byte {
	a := [8]byte{
		byte(x >> 56),
		byte(x >> 48),
		byte(x >> 40),
		byte(x >> 32),
		byte(x >> 24),
		byte(x >> 16),
		byte(x >> 8),
		byte(x),
	}
	return append(b, a[:]...)
}

func readUint64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
}

func update(crc uint64, tab *Table, p []byte) uint64 {
	buildSlicing8TablesOnce()
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

// Update 返回将切片 p 中的字节 添加到 crc 的结果。
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

// 返回数据 data 使用 table 代表的多项式计算出 CRC-64 校验和。
func Checksum(data []byte, tab *Table) uint64 { return update(0, tab, data) }

// tableSum returns the ISO checksum of table t.
func tableSum(t *Table) uint64 {
	var a [2048]byte
	b := a[:0]
	if t != nil {
		for _, x := range t {
			b = appendUint64(b, x)
		}
	}
	return Checksum(b, MakeTable(ISO))
}
