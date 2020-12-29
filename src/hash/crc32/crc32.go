// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// crc32 包实现了32 位循环冗余校验或者 CRC-32 校验和算法。 参见：https://en.wikipedia.org/wiki/Cyclic_redundancy_check。
//
// 多项式以 LSB-first 的形式表示，也称为反向表示。
// 参见：https://en.wikipedia.org/wiki/Mathematics_of_cyclic_redundancy_checks#Reversed_representations_and_reciprocal_polynomials
package crc32

import (
	"errors"
	"hash"
	"sync"
)

// CRC-32 校验和的字节长度
const Size = 4

// 预定义的多项式
const (
	// 到目前为止，IEEE 是最常见的 CRC-32 多项式。
	// 用于 以太网（IEEE 802.3）、 v.42、fddi、gzip、zip、pgn等。
	IEEE = 0xedb88320

	// Castagnoli 多项式，用在 iSCSI. 有比 IEEE 更好的错误探测特性。
	// https://dx.doi.org/10.1109/26.231911
	Castagnoli = 0x82f63b78

	// Koopman 多项式。错误探测特性也好于 IEEE。
	// https://dx.doi.org/10.1109/DSN.2002.1028931
	Koopman = 0xeb31d82e
)

// Table 是一个长度为 256 的 uint32 切片，代表一个用于高效运作的多项式。
type Table [256]uint32

// This file makes use of functions implemented in architecture-specific files.
// The interface that they implement is as follows:
//
//    // archAvailableIEEE reports whether an architecture-specific CRC32-IEEE
//    // algorithm is available.
//    archAvailableIEEE() bool
//
//    // archInitIEEE initializes the architecture-specific CRC3-IEEE algorithm.
//    // It can only be called if archAvailableIEEE() returns true.
//    archInitIEEE()
//
//    // archUpdateIEEE updates the given CRC32-IEEE. It can only be called if
//    // archInitIEEE() was previously called.
//    archUpdateIEEE(crc uint32, p []byte) uint32
//
//    // archAvailableCastagnoli reports whether an architecture-specific
//    // CRC32-C algorithm is available.
//    archAvailableCastagnoli() bool
//
//    // archInitCastagnoli initializes the architecture-specific CRC32-C
//    // algorithm. It can only be called if archAvailableCastagnoli() returns
//    // true.
//    archInitCastagnoli()
//
//    // archUpdateCastagnoli updates the given CRC32-C. It can only be called
//    // if archInitCastagnoli() was previously called.
//    archUpdateCastagnoli(crc uint32, p []byte) uint32

// castagnoliTable points to a lazily initialized Table for the Castagnoli
// polynomial. MakeTable will always return this value when asked to make a
// Castagnoli table so we can compare against it to find when the caller is
// using this polynomial.
var castagnoliTable *Table
var castagnoliTable8 *slicing8Table
var castagnoliArchImpl bool
var updateCastagnoli func(crc uint32, p []byte) uint32
var castagnoliOnce sync.Once

func castagnoliInit() {
	castagnoliTable = simpleMakeTable(Castagnoli)
	castagnoliArchImpl = archAvailableCastagnoli()

	if castagnoliArchImpl {
		archInitCastagnoli()
		updateCastagnoli = archUpdateCastagnoli
	} else {
		// Initialize the slicing-by-8 table.
		castagnoliTable8 = slicingMakeTable(Castagnoli)
		updateCastagnoli = func(crc uint32, p []byte) uint32 {
			return slicingUpdate(crc, castagnoliTable8, p)
		}
	}
}

// IEEETable 是 IEEE 多项式对应的 table
var IEEETable = simpleMakeTable(IEEE)

// ieeeTable8 is the slicing8Table for IEEE
var ieeeTable8 *slicing8Table
var ieeeArchImpl bool
var updateIEEE func(crc uint32, p []byte) uint32
var ieeeOnce sync.Once

func ieeeInit() {
	ieeeArchImpl = archAvailableIEEE()

	if ieeeArchImpl {
		archInitIEEE()
		updateIEEE = archUpdateIEEE
	} else {
		// Initialize the slicing-by-8 table.
		ieeeTable8 = slicingMakeTable(IEEE)
		updateIEEE = func(crc uint32, p []byte) uint32 {
			return slicingUpdate(crc, ieeeTable8, p)
		}
	}
}

// MakeTable 返回根据指定多项式构造的 table。
// 该表的内容不得修改
func MakeTable(poly uint32) *Table {
	switch poly {
	case IEEE:
		ieeeOnce.Do(ieeeInit)
		return IEEETable
	case Castagnoli:
		castagnoliOnce.Do(castagnoliInit)
		return castagnoliTable
	}
	return simpleMakeTable(poly)
}

// digest represents the partial evaluation of a checksum.
type digest struct {
	crc uint32
	tab *Table
}

// New 创建一个 使用 table 表示的多项式计算 CRC-32 校验和的 hash.Hash32。
// 它的 Sum 方法将按照 big-endian 字节顺序排列值。返回的 Hash32 也实现了
// encoding.BinaryMarshaler 和 encoding.BinaryUnmarshaler 来封装和取消封装hash的内部状态。
func New(tab *Table) hash.Hash32 {
	if tab == IEEETable {
		ieeeOnce.Do(ieeeInit)
	}
	return &digest{0, tab}
}

// NewIEEE 创建一个使用 IEEE 多项式计算 CRC-32 校验和的 hash.Hash32。
// 它的 Sum 方法将按照 big-endian 字节顺序排列值。 返回的 Hash32 也实现了
// encoding.BinaryMarshaler 和 encoding.BinaryUnmarshaler 来封装和取消封装hash的内部状态。

func NewIEEE() hash.Hash32 { return New(IEEETable) }

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return 1 }

func (d *digest) Reset() { d.crc = 0 }

const (
	magic         = "crc\x01"
	marshaledSize = len(magic) + 4 + 4
)

func (d *digest) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, marshaledSize)
	b = append(b, magic...)
	b = appendUint32(b, tableSum(d.tab))
	b = appendUint32(b, d.crc)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic) || string(b[:len(magic)]) != magic {
		return errors.New("hash/crc32: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("hash/crc32: invalid hash state size")
	}
	if tableSum(d.tab) != readUint32(b[4:]) {
		return errors.New("hash/crc32: tables do not match")
	}
	d.crc = readUint32(b[8:])
	return nil
}

func appendUint32(b []byte, x uint32) []byte {
	a := [4]byte{
		byte(x >> 24),
		byte(x >> 16),
		byte(x >> 8),
		byte(x),
	}
	return append(b, a[:]...)
}

func readUint32(b []byte) uint32 {
	_ = b[3]
	return uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24
}

// Update 返回将切片 p 中的字节 添加到 crc 的结果。
func Update(crc uint32, tab *Table, p []byte) uint32 {
	switch tab {
	case castagnoliTable:
		return updateCastagnoli(crc, p)
	case IEEETable:
		// Unfortunately, because IEEETable is exported, IEEE may be used without a
		// call to MakeTable. We have to make sure it gets initialized in that case.
		ieeeOnce.Do(ieeeInit)
		return updateIEEE(crc, p)
	default:
		return simpleUpdate(crc, tab, p)
	}
}

func (d *digest) Write(p []byte) (n int, err error) {
	switch d.tab {
	case castagnoliTable:
		d.crc = updateCastagnoli(d.crc, p)
	case IEEETable:
		// We only create digest objects through New() which takes care of
		// initialization in this case.
		d.crc = updateIEEE(d.crc, p)
	default:
		d.crc = simpleUpdate(d.crc, d.tab, p)
	}
	return len(p), nil
}

func (d *digest) Sum32() uint32 { return d.crc }

func (d *digest) Sum(in []byte) []byte {
	s := d.Sum32()
	return append(in, byte(s>>24), byte(s>>16), byte(s>>8), byte(s))
}

// 返回数据 data 使用 table 代表的多项式计算出 CRC-32 校验和。
func Checksum(data []byte, tab *Table) uint32 { return Update(0, tab, data) }

// 返回数据 data 使用 IEEE 多项式计算出 CRC-32 校验和。
func ChecksumIEEE(data []byte) uint32 {
	ieeeOnce.Do(ieeeInit)
	return updateIEEE(0, data)
}

// tableSum returns the IEEE checksum of table t.
func tableSum(t *Table) uint32 {
	var a [1024]byte
	b := a[:0]
	if t != nil {
		for _, x := range t {
			b = appendUint32(b, x)
		}
	}
	return ChecksumIEEE(b)
}
