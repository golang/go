// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import "internal/cpu"

func ieeeUpdateCLMUL(crc uint32, p []byte) uint32

func castagnoliUpdateCLMUL(crc uint32, p []byte) uint32

func archAvailableIEEE() bool {
	return cpu.RISCV64.HasZbc
}

var archIeeeTable8 *slicing8Table

func archInitIEEE() {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for IEEE not available")
	}
	// We still use slicing-by-8 for small buffers.
	archIeeeTable8 = slicingMakeTable(IEEE)
}

func archUpdateIEEE(crc uint32, p []byte) uint32 {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for IEEE not available")
	}

	if len(p) >= 16 {
		left := len(p) & 15
		do := len(p) - left
		crc = ^ieeeUpdateCLMUL(^crc, p[:do])
		p = p[do:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archIeeeTable8, p)
}

func archAvailableCastagnoli() bool {
	return cpu.RISCV64.HasZbc
}

var archCastagnoliTable8 *slicing8Table

func archInitCastagnoli() {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for Castagnoli not available")
	}
	// We still use slicing-by-8 for small buffers.
	archCastagnoliTable8 = slicingMakeTable(Castagnoli)
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for Castagnoli not available")
	}

	if len(p) >= 16 {
		left := len(p) & 15
		do := len(p) - left
		crc = ^castagnoliUpdateCLMUL(^crc, p[:do])
		p = p[do:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archCastagnoliTable8, p)
}
