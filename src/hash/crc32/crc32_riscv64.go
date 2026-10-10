// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import "internal/cpu"

//go:noescape
func ieeeUpdateCLMUL(crc uint32, tab *Table, p []byte) uint32

//go:noescape
func castagnoliUpdateCLMUL(crc uint32, tab *Table, p []byte) uint32

func archAvailableIEEE() bool {
	return cpu.RISCV64.HasZbc
}

var archIeeeTable8 *slicing8Table

func archInitIEEE() {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for IEEE not available")
	}
	// The slicing-by-8 table is used for byte-at-a-time alignment (inside the
	// CLMUL routine) and for the tail that does not form a complete 16-byte
	// block. It is also the fallback for inputs shorter than 16 bytes.
	archIeeeTable8 = slicingMakeTable(IEEE)
}

func archUpdateIEEE(crc uint32, p []byte) uint32 {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for IEEE not available")
	}

	// The CLMUL routine aligns a misaligned pointer to a 16-byte boundary
	// one byte at a time (table-based) before entering the fold loop, so all
	// LD loads inside it are aligned and it is safe on cores that trap on
	// unaligned accesses. Give it all complete 16-byte blocks.
	if len(p) >= 16 {
		aligned := len(p) &^ 15
		crc = ^ieeeUpdateCLMUL(^crc, &archIeeeTable8[0], p[:aligned])
		p = p[aligned:]
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
	// The slicing-by-8 table is used for byte-at-a-time alignment (inside the
	// CLMUL routine) and for the tail that does not form a complete 16-byte
	// block. It is also the fallback for inputs shorter than 16 bytes.
	archCastagnoliTable8 = slicingMakeTable(Castagnoli)
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !cpu.RISCV64.HasZbc {
		panic("arch-specific zbc instruction for Castagnoli not available")
	}

	// Same as archUpdateIEEE: the CLMUL routine aligns a misaligned pointer
	// byte-at-a-time before the fold loop.
	if len(p) >= 16 {
		aligned := len(p) &^ 15
		crc = ^castagnoliUpdateCLMUL(^crc, &archCastagnoliTable8[0], p[:aligned])
		p = p[aligned:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archCastagnoliTable8, p)
}
