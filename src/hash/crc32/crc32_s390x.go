// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

const (
	vxMinLen    = 64
	vxAlignMask = 15 // align to 16 bytes
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

// updateCastagnoli calculates the checksum of p using
// vectorizedCastagnoli if possible and falling back onto
// genericCastagnoli as needed.
func updateCastagnoli(crc uint32, p []byte) uint32 {
	// Use vectorized function if vector facility is available and
	// data length is above threshold.
	if hasVX && len(p) >= vxMinLen {
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

// updateIEEE calculates the checksum of p using vectorizedIEEE if
// possible and falling back onto genericIEEE as needed.
func updateIEEE(crc uint32, p []byte) uint32 {
	// Use vectorized function if vector facility is available and
	// data length is above threshold.
	if hasVX && len(p) >= vxMinLen {
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
