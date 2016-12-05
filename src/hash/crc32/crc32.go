// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crc32 implements the 32-bit cyclic redundancy check, or CRC-32,
// checksum. See http://en.wikipedia.org/wiki/Cyclic_redundancy_check for
// information.
//
// Polynomials are represented in LSB-first form also known as reversed representation.
//
// See http://en.wikipedia.org/wiki/Mathematics_of_cyclic_redundancy_checks#Reversed_representations_and_reciprocal_polynomials
// for information.
package crc32

import (
	"hash"
	"sync"
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
	// http://dx.doi.org/10.1109/26.231911
	Castagnoli = 0x82f63b78

	// Koopman's polynomial.
	// Also has better error detection characteristics than IEEE.
	// http://dx.doi.org/10.1109/DSN.2002.1028931
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

// IEEETable is the table for the IEEE polynomial.
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

// MakeTable returns a Table constructed from the specified polynomial.
// The contents of this Table must not be modified.
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

// New creates a new hash.Hash32 computing the CRC-32 checksum
// using the polynomial represented by the Table.
// Its Sum method will lay the value out in big-endian byte order.
func New(tab *Table) hash.Hash32 {
	if tab == IEEETable {
		ieeeOnce.Do(ieeeInit)
	}
	return &digest{0, tab}
}

// NewIEEE creates a new hash.Hash32 computing the CRC-32 checksum
// using the IEEE polynomial.
// Its Sum method will lay the value out in big-endian byte order.
func NewIEEE() hash.Hash32 { return New(IEEETable) }

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return 1 }

func (d *digest) Reset() { d.crc = 0 }

// Update returns the result of adding the bytes in p to the crc.
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

// Checksum returns the CRC-32 checksum of data
// using the polynomial represented by the Table.
func Checksum(data []byte, tab *Table) uint32 { return Update(0, tab, data) }

// ChecksumIEEE returns the CRC-32 checksum of data
// using the IEEE polynomial.
func ChecksumIEEE(data []byte) uint32 {
	ieeeOnce.Do(ieeeInit)
	return updateIEEE(0, data)
}
