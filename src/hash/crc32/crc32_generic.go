// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm arm64 mips64 mips64le ppc64 ppc64le

package crc32

// The file contains the generic version of updateCastagnoli which just calls
// the software implementation.

func updateCastagnoli(crc uint32, p []byte) uint32 {
	return update(crc, castagnoliTable, p)
}

func updateIEEE(crc uint32, p []byte) uint32 {
	// only use slicing-by-8 when input is >= 4KB
	if len(p) >= 4096 {
		ieeeTable8Once.Do(func() {
			ieeeTable8 = makeTable8(IEEE)
		})
		return updateSlicingBy8(crc, ieeeTable8, p)
	}
	return update(crc, IEEETable, p)
}
