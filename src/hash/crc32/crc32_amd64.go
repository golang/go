// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// AMD64-specific hardware-assisted CRC32 algorithms. See crc32.go for a
// description of the interface that each architecture-specific file
// implements.

package crc32

import (
	"internal/cpu"
	"unsafe"
)

// Offset into internal/cpu records for use in assembly.
const (
	offsetX86HasAVX512VPCLMULQDQL = unsafe.Offsetof(cpu.X86.HasAVX512VPCLMULQDQ)
)

// This file contains the code to call the SSE 4.2 version of the Castagnoli
// and IEEE CRC.

// castagnoliSSE42 is defined in crc32_amd64.s and uses the SSE 4.2 CRC32
// instruction.
//
//go:noescape
func castagnoliSSE42(crc uint32, p []byte) uint32

// castagnoliSSE42Triple is defined in crc32_amd64.s and uses the SSE 4.2 CRC32
// instruction.
//
//go:noescape
func castagnoliSSE42Triple(
	crcA, crcB, crcC uint32,
	a, b, c []byte,
	rounds uint32,
) (retA uint32, retB uint32, retC uint32)

// ieeeCLMUL is defined in crc_amd64.s and uses the PCLMULQDQ
// instruction as well as SSE 4.1.
//
//go:noescape
func ieeeCLMUL(crc uint32, p []byte) uint32

const castagnoliK1 = 168
const castagnoliK2 = 1344

type sse42Table [4]Table

var castagnoliSSE42TableK1 *sse42Table
var castagnoliSSE42TableK2 *sse42Table

func archAvailableCastagnoli() bool {
	return cpu.X86.HasSSE42
}

func archInitCastagnoli() {
	if !cpu.X86.HasSSE42 {
		panic("arch-specific Castagnoli not available")
	}
	castagnoliSSE42TableK1 = new(sse42Table)
	castagnoliSSE42TableK2 = new(sse42Table)
	// See description in updateCastagnoli.
	//    t[0][i] = CRC(i000, O)
	//    t[1][i] = CRC(0i00, O)
	//    t[2][i] = CRC(00i0, O)
	//    t[3][i] = CRC(000i, O)
	// where O is a sequence of K zeros.
	var tmp [castagnoliK2]byte
	for b := 0; b < 4; b++ {
		for i := 0; i < 256; i++ {
			val := uint32(i) << uint32(b*8)
			castagnoliSSE42TableK1[b][i] = castagnoliSSE42(val, tmp[:castagnoliK1])
			castagnoliSSE42TableK2[b][i] = castagnoliSSE42(val, tmp[:])
		}
	}
}

// castagnoliShift computes the CRC32-C of K1 or K2 zeroes (depending on the
// table given) with the given initial crc value. This corresponds to
// CRC(crc, O) in the description in updateCastagnoli.
func castagnoliShift(table *sse42Table, crc uint32) uint32 {
	return table[3][crc>>24] ^
		table[2][(crc>>16)&0xFF] ^
		table[1][(crc>>8)&0xFF] ^
		table[0][crc&0xFF]
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !cpu.X86.HasSSE42 {
		panic("not available")
	}

	// This method is inspired from the algorithm in Intel's white paper:
	//    "Fast CRC Computation for iSCSI Polynomial Using CRC32 Instruction"
	// The same strategy of splitting the buffer in three is used but the
	// combining calculation is different; the complete derivation is explained
	// below.
	//
	// -- The basic idea --
	//
	// The CRC32 instruction (available in SSE4.2) can process 8 bytes at a
	// time. In recent Intel architectures the instruction takes 3 cycles;
	// however the processor can pipeline up to three instructions if they
	// don't depend on each other.
	//
	// Roughly this means that we can process three buffers in about the same
	// time we can process one buffer.
	//
	// The idea is then to split the buffer in three, CRC the three pieces
	// separately and then combine the results.
	//
	// Combining the results requires precomputed tables, so we must choose a
	// fixed buffer length to optimize. The longer the length, the faster; but
	// only buffers longer than this length will use the optimization. We choose
	// two cutoffs and compute tables for both:
	//  - one around 512: 168*3=504
	//  - one around 4KB: 1344*3=4032
	//
	// -- The nitty gritty --
	//
	// Let CRC(I, X) be the non-inverted CRC32-C of the sequence X (with
	// initial non-inverted CRC I). This function has the following properties:
	//   (a) CRC(I, AB) = CRC(CRC(I, A), B)
	//   (b) CRC(I, A xor B) = CRC(I, A) xor CRC(0, B)
	//
	// Say we want to compute CRC(I, ABC) where A, B, C are three sequences of
	// K bytes each, where K is a fixed constant. Let O be the sequence of K zero
	// bytes.
	//
	// CRC(I, ABC) = CRC(I, ABO xor C)
	//             = CRC(I, ABO) xor CRC(0, C)
	//             = CRC(CRC(I, AB), O) xor CRC(0, C)
	//             = CRC(CRC(I, AO xor B), O) xor CRC(0, C)
	//             = CRC(CRC(I, AO) xor CRC(0, B), O) xor CRC(0, C)
	//             = CRC(CRC(CRC(I, A), O) xor CRC(0, B), O) xor CRC(0, C)
	//
	// The castagnoliSSE42Triple function can compute CRC(I, A), CRC(0, B),
	// and CRC(0, C) efficiently.  We just need to find a way to quickly compute
	// CRC(uvwx, O) given a 4-byte initial value uvwx. We can precompute these
	// values; since we can't have a 32-bit table, we break it up into four
	// 8-bit tables:
	//
	//    CRC(uvwx, O) = CRC(u000, O) xor
	//                   CRC(0v00, O) xor
	//                   CRC(00w0, O) xor
	//                   CRC(000x, O)
	//
	// We can compute tables corresponding to the four terms for all 8-bit
	// values.

	crc = ^crc

	// If a buffer is long enough to use the optimization, process the first few
	// bytes to align the buffer to an 8 byte boundary (if necessary).
	if len(p) >= castagnoliK1*3 {
		delta := int(uintptr(unsafe.Pointer(&p[0])) & 7)
		if delta != 0 {
			delta = 8 - delta
			crc = castagnoliSSE42(crc, p[:delta])
			p = p[delta:]
		}
	}

	// Process 3*K2 at a time.
	for len(p) >= castagnoliK2*3 {
		// Compute CRC(I, A), CRC(0, B), and CRC(0, C).
		crcA, crcB, crcC := castagnoliSSE42Triple(
			crc, 0, 0,
			p, p[castagnoliK2:], p[castagnoliK2*2:],
			castagnoliK2/24)

		// CRC(I, AB) = CRC(CRC(I, A), O) xor CRC(0, B)
		crcAB := castagnoliShift(castagnoliSSE42TableK2, crcA) ^ crcB
		// CRC(I, ABC) = CRC(CRC(I, AB), O) xor CRC(0, C)
		crc = castagnoliShift(castagnoliSSE42TableK2, crcAB) ^ crcC
		p = p[castagnoliK2*3:]
	}

	// Process 3*K1 at a time.
	for len(p) >= castagnoliK1*3 {
		// Compute CRC(I, A), CRC(0, B), and CRC(0, C).
		crcA, crcB, crcC := castagnoliSSE42Triple(
			crc, 0, 0,
			p, p[castagnoliK1:], p[castagnoliK1*2:],
			castagnoliK1/24)

		// CRC(I, AB) = CRC(CRC(I, A), O) xor CRC(0, B)
		crcAB := castagnoliShift(castagnoliSSE42TableK1, crcA) ^ crcB
		// CRC(I, ABC) = CRC(CRC(I, AB), O) xor CRC(0, C)
		crc = castagnoliShift(castagnoliSSE42TableK1, crcAB) ^ crcC
		p = p[castagnoliK1*3:]
	}

	// Use the simple implementation for what's left.
	crc = castagnoliSSE42(crc, p)
	return ^crc
}

func archAvailableIEEE() bool {
	return cpu.X86.HasPCLMULQDQ && cpu.X86.HasSSE41
}

var archIeeeTable8 *slicing8Table

func archInitIEEE() {
	if !cpu.X86.HasPCLMULQDQ || !cpu.X86.HasSSE41 {
		panic("not available")
	}
	// We still use slicing-by-8 for small buffers.
	archIeeeTable8 = slicingMakeTable(IEEE)
}

func archUpdateIEEE(crc uint32, p []byte) uint32 {
	if !cpu.X86.HasPCLMULQDQ || !cpu.X86.HasSSE41 {
		panic("not available")
	}

	if len(p) >= 64 {
		left := len(p) & 15
		do := len(p) - left
		crc = ^ieeeCLMUL(^crc, p[:do])
		p = p[do:]
	}
	if len(p) == 0 {
		return crc
	}
	return slicingUpdate(crc, archIeeeTable8, p)
}
