// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ARM64-specific hardware-assisted CRC32 algorithms. See crc32.go for a
// description of the interface that each architecture-specific file
// implements.

package crc32

import "internal/cpu"

// castagnoliUpdate does not retain p, so callers' buffers can stay on the stack.
//
//go:noescape
func castagnoliUpdate(crc uint32, p []byte) uint32

func ieeeUpdate(crc uint32, p []byte) uint32

// With GOEXPERIMENT=simd, castagnoliUpdatePMULL folds from 256 bytes.
const pmullMinLength = 256

// pmullFold reports whether archUpdateCastagnoli folds an n-byte input.
func pmullFold(n int) bool {
	return n >= pmullMinLength && usePMULL
}

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

	if pmullFold(len(p)) {
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
