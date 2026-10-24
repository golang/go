// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ARM64-specific hardware-assisted CRC32 algorithms. See crc32.go for a
// description of the interface that each architecture-specific file
// implements.

package crc32

import "internal/cpu"

func castagnoliUpdate(crc uint32, p []byte) uint32

// castagnoliUpdatePMULL is castagnoliUpdate for CPUs that also have PMULL.
//
//go:noescape
func castagnoliUpdatePMULL(crc uint32, p []byte) uint32
func ieeeUpdate(crc uint32, p []byte) uint32

// castagnoliUpdatePMULL only folds from 256 bytes.
const pmullMinLength = 256

// SHA3 excludes cores like Neoverse N1, where the fold is slower than CRC32C.
var usePMULL = cpu.ARM64.HasPMULL && cpu.ARM64.HasSHA3

func archAvailableCastagnoli() bool {
	return cpu.ARM64.HasCRC32
}

func archInitCastagnoli() {
	if !cpu.ARM64.HasCRC32 {
		panic("arch-specific crc32 instruction for Castagnoli not available")
	}
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !cpu.ARM64.HasCRC32 {
		panic("arch-specific crc32 instruction for Castagnoli not available")
	}

	if len(p) >= pmullMinLength && usePMULL {
		return ^castagnoliUpdatePMULL(^crc, p)
	}
	return ^castagnoliUpdate(^crc, p)
}

func archAvailableIEEE() bool {
	return cpu.ARM64.HasCRC32
}

func archInitIEEE() {
	if !cpu.ARM64.HasCRC32 {
		panic("arch-specific crc32 instruction for IEEE not available")
	}
}

func archUpdateIEEE(crc uint32, p []byte) uint32 {
	if !cpu.ARM64.HasCRC32 {
		panic("arch-specific crc32 instruction for IEEE not available")
	}

	return ^ieeeUpdate(^crc, p)
}
