// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crc32 implements the 32-bit cyclic redundancy check, or CRC-32,
// checksum. See https://en.wikipedia.org/wiki/Cyclic_redundancy_check for
// information.
//
// Polynomials are represented in LSB-first form also known as reversed representation.
//
// See https://en.wikipedia.org/wiki/Mathematics_of_cyclic_redundancy_checks#Reversed_representations_and_reciprocal_polynomials
// for information.
package crc32

import (
	"errors"
	"hash"
	"internal/byteorder"
	"sync"
	"sync/atomic"
)

// The size of a CRC-32 checksum in bytes.
const Size = 4

// Predefined polynomials.
const (
	// IEEE is by far and away the most common CRC-32 polynomial.
	// Used by ethernet (IEEE 802.3), v.42, fddi, gzip, zip, png, ...
	IEEE = 0xedb88320

	// Castagnoli's polynomial, used in iSCSI.
	// Has better error detection characteristics than IEEE.
	// https://dx.doi.org/10.1109/26.231911
	Castagnoli = 0x82f63b78

	// Koopman's polynomial.
	// Also has better error detection characteristics than IEEE.
	// https://dx.doi.org/10.1109/DSN.2002.1028931
	Koopman = 0xeb31d82e
)

// Table is a 256-word table representing the polynomial for efficient processing.
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
var updateCastagnoli func(crc uint32, p []byte) uint32
var castagnoliOnce sync.Once
var haveCastagnoli atomic.Bool

func castagnoliInit() {
	castagnoliTable = simpleMakeTable(Castagnoli)

	if archAvailableCastagnoli() {
		archInitCastagnoli()
		updateCastagnoli = archUpdateCastagnoli
	} else {
		// Initialize the slicing-by-8 table.
		castagnoliTable8 = slicingMakeTable(Castagnoli)
		updateCastagnoli = func(crc uint32, p []byte) uint32 {
			return slicingUpdate(crc, castagnoliTable8, p)
		}
	}

	haveCastagnoli.Store(true)
}

// IEEETable is the table for the [IEEE] polynomial.
var IEEETable = simpleMakeTable(IEEE)

// ieeeTable8 is the slicing8Table for IEEE
var ieeeTable8 *slicing8Table
var updateIEEE func(crc uint32, p []byte) uint32
var ieeeOnce sync.Once

func ieeeInit() {
	if archAvailableIEEE() {
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

// MakeTable returns a [Table] constructed from the specified polynomial.
// The contents of this [Table] must not be modified.
func MakeTable(poly uint32) *Table {
	switch poly {
	case IEEE:
		ieeeOnce.Do(ieeeInit)
		return IEEETable
	case Castagnoli:
		castagnoliOnce.Do(castagnoliInit)
		return castagnoliTable
	default:
		return simpleMakeTable(poly)
	}
}

// digest represents the partial evaluation of a checksum.
type digest struct {
	crc uint32
	tab *Table
}

// New creates a new [hash.Hash32] computing the CRC-32 checksum using the
// polynomial represented by the [Table]. Its Sum method will lay the
// value out in big-endian byte order. The returned Hash32 also
// implements [encoding.BinaryMarshaler] and [encoding.BinaryUnmarshaler] to
// marshal and unmarshal the internal state of the hash.
func New(tab *Table) hash.Hash32 {
	if tab == IEEETable {
		ieeeOnce.Do(ieeeInit)
	}
	return &digest{0, tab}
}

// NewIEEE creates a new [hash.Hash32] computing the CRC-32 checksum using
// the [IEEE] polynomial. Its Sum method will lay the value out in
// big-endian byte order. The returned Hash32 also implements
// [encoding.BinaryMarshaler] and [encoding.BinaryUnmarshaler] to marshal
// and unmarshal the internal state of the hash.
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
	b = byteorder.BeAppendUint32(b, tableSum(d.tab))
	b = byteorder.BeAppendUint32(b, d.crc)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic) || string(b[:len(magic)]) != magic {
		return errors.New("hash/crc32: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("hash/crc32: invalid hash state size")
	}
	if tableSum(d.tab) != byteorder.BeUint32(b[4:]) {
		return errors.New("hash/crc32: tables do not match")
	}
	d.crc = byteorder.BeUint32(b[8:])
	return nil
}

func update(crc uint32, tab *Table, p []byte, checkInitIEEE bool) uint32 {
	switch {
	case haveCastagnoli.Load() && tab == castagnoliTable:
		return updateCastagnoli(crc, p)
	case tab == IEEETable:
		if checkInitIEEE {
			ieeeOnce.Do(ieeeInit)
		}
		return updateIEEE(crc, p)
	default:
		return simpleUpdate(crc, tab, p)
	}
}

// Update returns the result of adding the bytes in p to the crc.
func Update(crc uint32, tab *Table, p []byte) uint32 {
	// Unfortunately, because IEEETable is exported, IEEE may be used without a
	// call to MakeTable. We have to make sure it gets initialized in that case.
	return update(crc, tab, p, true)
}

func (d *digest) Write(p []byte) (n int, err error) {
	// We only create digest objects through New() which takes care of
	// initialization in this case.
	d.crc = update(d.crc, d.tab, p, false)
	return len(p), nil
}

func (d *digest) Sum32() uint32 { return d.crc }

func (d *digest) Sum(in []byte) []byte {
	s := d.Sum32()
	return append(in, byte(s>>24), byte(s>>16), byte(s>>8), byte(s))
}

// Checksum returns the CRC-32 checksum of data
// using the polynomial represented by the [Table].
func Checksum(data []byte, tab *Table) uint32 { return Update(0, tab, data) }

// ChecksumIEEE returns the CRC-32 checksum of data
// using the [IEEE] polynomial.
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
			b = byteorder.BeAppendUint32(b, x)
		}
	}
	return ChecksumIEEE(b)
}
