// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import "internal/cpu"

const (
	vxMinLen    = 64
	vxAlignMask = 15 // align to 16 bytes
)

// hasVX reports whether the machine has the z/Architecture
// vector facility installed and enabled.
var hasVX = cpu.S390X.HasVX

// vectorizedCastagnoli implements CRC32 using vector instructions.
// It is defined in crc32_s390x.s.
//
//go:noescape
func vectorizedCastagnoli(crc uint32, p []byte) uint32

// vectorizedIEEE implements CRC32 using vector instructions.
// It is defined in crc32_s390x.s.
//
//go:noescape
func vectorizedIEEE(crc uint32, p []byte) uint32

func archAvailableCastagnoli() bool {
	return hasVX
}

var archCastagnoliTable8 *slicing8Table

func archInitCastagnoli() {
	if !hasVX {
		panic("not available")
	}
	// We still use slicing-by-8 for small buffers.
	archCastagnoliTable8 = slicingMakeTable(Castagnoli)
}

// archUpdateCastagnoli calculates the checksum of p using
// vectorizedCastagnoli.
func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !hasVX {
		panic("not available")
	}
	// Use vectorized function if data length is above threshold.
	if len(p) >= vxMinLen {
		aligned := len(p) & ^vxAlignMask
		crc = vectorizedCastagnoli(crc, p[:aligned])
		p = p[aligned:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archCastagnoliTable8, p)
}

func archAvailableIEEE() bool {
	return hasVX
}

var archIeeeTable8 *slicing8Table

func archInitIEEE() {
	if !hasVX {
		panic("not available")
	}
	// We still use slicing-by-8 for small buffers.
	archIeeeTable8 = slicingMakeTable(IEEE)
}

// archUpdateIEEE calculates the checksum of p using vectorizedIEEE.
func archUpdateIEEE(crc uint32, p []byte) uint32 {
	if !hasVX {
		panic("not available")
	}
	// Use vectorized function if data length is above threshold.
	if len(p) >= vxMinLen {
		aligned := len(p) & ^vxAlignMask
		crc = vectorizedIEEE(crc, p[:aligned])
		p = p[aligned:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archIeeeTable8, p)
}
