// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"unsafe"
)

const (
	vxMinLen    = 64
	vxAlignment = 16
	vxAlignMask = vxAlignment - 1
)

// hasVectorFacility reports whether the machine has the z/Architecture
// vector facility installed and enabled.
func hasVectorFacility() bool

var hasVX = hasVectorFacility()

// vectorizedCastagnoli implements CRC32 using vector instructions.
// It is defined in crc32_s390x.s.
//go:noescape
func vectorizedCastagnoli(crc uint32, p []byte) uint32

// vectorizedIEEE implements CRC32 using vector instructions.
// It is defined in crc32_s390x.s.
//go:noescape
func vectorizedIEEE(crc uint32, p []byte) uint32

func genericCastagnoli(crc uint32, p []byte) uint32 {
	// Use slicing-by-8 on larger inputs.
	if len(p) >= sliceBy8Cutoff {
		return updateSlicingBy8(crc, castagnoliTable8, p)
	}
	return update(crc, castagnoliTable, p)
}

func genericIEEE(crc uint32, p []byte) uint32 {
	// Use slicing-by-8 on larger inputs.
	if len(p) >= sliceBy8Cutoff {
		ieeeTable8Once.Do(func() {
			ieeeTable8 = makeTable8(IEEE)
		})
		return updateSlicingBy8(crc, ieeeTable8, p)
	}
	return update(crc, IEEETable, p)
}

// updateCastagnoli calculates the checksum of p using genericCastagnoli to
// align the data appropriately for vectorCastagnoli. It avoids using
// vectorCastagnoli entirely if the length of p is less than or equal to
// vxMinLen.
func updateCastagnoli(crc uint32, p []byte) uint32 {
	// Use vectorized function if vector facility is available and
	// data length is above threshold.
	if hasVX && len(p) > vxMinLen {
		pAddr := uintptr(unsafe.Pointer(&p[0]))
		if pAddr&vxAlignMask != 0 {
			prealign := vxAlignment - int(pAddr&vxAlignMask)
			crc = genericCastagnoli(crc, p[:prealign])
			p = p[prealign:]
		}
		aligned := len(p) & ^vxAlignMask
		crc = vectorizedCastagnoli(crc, p[:aligned])
		p = p[aligned:]
		// process remaining data
		if len(p) > 0 {
			crc = genericCastagnoli(crc, p)
		}
		return crc
	}
	return genericCastagnoli(crc, p)
}

// updateIEEE calculates the checksum of p using genericIEEE to align the data
// appropriately for vectorIEEE. It avoids using vectorIEEE entirely if the length
// of p is less than or equal to vxMinLen.
func updateIEEE(crc uint32, p []byte) uint32 {
	// Use vectorized function if vector facility is available and
	// data length is above threshold.
	if hasVX && len(p) > vxMinLen {
		pAddr := uintptr(unsafe.Pointer(&p[0]))
		if pAddr&vxAlignMask != 0 {
			prealign := vxAlignment - int(pAddr&vxAlignMask)
			crc = genericIEEE(crc, p[:prealign])
			p = p[prealign:]
		}
		aligned := len(p) & ^vxAlignMask
		crc = vectorizedIEEE(crc, p[:aligned])
		p = p[aligned:]
		// process remaining data
		if len(p) > 0 {
			crc = genericIEEE(crc, p)
		}
		return crc
	}
	return genericIEEE(crc, p)
}
