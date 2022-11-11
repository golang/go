// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"unsafe"
)

const (
	vecMinLen    = 16
	vecAlignMask = 15 // align to 16 bytes
	crcIEEE      = 1
	crcCast      = 2
)

//go:noescape
func ppc64SlicingUpdateBy8(crc uint32, table8 *slicing8Table, p []byte) uint32

// this function requires the buffer to be 16 byte aligned and > 16 bytes long.
//
//go:noescape
func vectorCrc32(crc uint32, poly uint32, p []byte) uint32

var archCastagnoliTable8 *slicing8Table

func archInitCastagnoli() {
	archCastagnoliTable8 = slicingMakeTable(Castagnoli)
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if len(p) >= 4*vecMinLen {
		// If not aligned then process the initial unaligned bytes

		if uint64(uintptr(unsafe.Pointer(&p[0])))&uint64(vecAlignMask) != 0 {
			align := uint64(uintptr(unsafe.Pointer(&p[0]))) & uint64(vecAlignMask)
			newlen := vecMinLen - align
			crc = ppc64SlicingUpdateBy8(crc, archCastagnoliTable8, p[:newlen])
			p = p[newlen:]
		}
		// p should be aligned now
		aligned := len(p) & ^vecAlignMask
		crc = vectorCrc32(crc, crcCast, p[:aligned])
		p = p[aligned:]
	}
	if len(p) == 0 {
		return crc
	}
	return ppc64SlicingUpdateBy8(crc, archCastagnoliTable8, p)
}

func archAvailableIEEE() bool {
	return true
}
func archAvailableCastagnoli() bool {
	return true
}

var archIeeeTable8 *slicing8Table

func archInitIEEE() {
	// We still use slicing-by-8 for small buffers.
	archIeeeTable8 = slicingMakeTable(IEEE)
}

// archUpdateIEEE calculates the checksum of p using vectorizedIEEE.
func archUpdateIEEE(crc uint32, p []byte) uint32 {

	// Check if vector code should be used.  If not aligned, then handle those
	// first up to the aligned bytes.

	if len(p) >= 4*vecMinLen {
		if uint64(uintptr(unsafe.Pointer(&p[0])))&uint64(vecAlignMask) != 0 {
			align := uint64(uintptr(unsafe.Pointer(&p[0]))) & uint64(vecAlignMask)
			newlen := vecMinLen - align
			crc = ppc64SlicingUpdateBy8(crc, archIeeeTable8, p[:newlen])
			p = p[newlen:]
		}
		aligned := len(p) & ^vecAlignMask
		crc = vectorCrc32(crc, crcIEEE, p[:aligned])
		p = p[aligned:]
	}
	if len(p) == 0 {
		return crc
	}
	return ppc64SlicingUpdateBy8(crc, archIeeeTable8, p)
}
