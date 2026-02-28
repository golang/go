// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

// 4K (smallest case) page size offset mask for PPC64.
#define PAGE_OFFSET 4095

// Likewise, the BC opcode is hard to read, and no extended
// mnemonics are offered for these forms.
#define BGELR_CR6 BC  4, CR6LT, (LR)
#define BEQLR     BC 12, CR0EQ, (LR)

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// R3 = a
	// R4 = b
	// R5 = size
	BR	memeqbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-17
	// R3 = a
	// R4 = b
	CMP	R3, R4
	BEQ	eq
	MOVD	8(R11), R5    // compiler stores size at offset 8 in the closure
	BR	memeqbody<>(SB)
eq:
	MOVD	$1, R3
	RET

// Do an efficient memequal for ppc64
// R3 = s1
// R4 = s2
// R5 = len
// On exit:
// R3 = return value
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R3, R8		// Move s1 into R8
	ADD	R5, R3, R9	// &s1[len(s1)]
	ADD	R5, R4, R10	// &s2[len(s2)]
	MOVD	$1, R11
	CMP	R5, $16		// Use GPR checks for check for len <= 16
	BLE	check0_16
	MOVD	$0, R3		// Assume no-match in case BGELR CR6 returns
	CMP	R5, $32		// Use overlapping VSX loads for len <= 32
	BLE	check17_32	// Do a pair of overlapping VSR compares
	CMP	R5, $64
	BLE	check33_64	// Hybrid check + overlap compare.

setup64:
	SRD	$6, R5, R6	// number of 64 byte chunks to compare
	MOVD	R6, CTR
	MOVD	$16, R14	// index for VSX loads and stores
	MOVD	$32, R15
	MOVD	$48, R16
	ANDCC	$0x3F, R5, R5	// len%64==0?

	PCALIGN $16
loop64:
	LXVD2X	(R8+R0), V0
	LXVD2X	(R4+R0), V1
	VCMPEQUBCC V0, V1, V2	// compare, setting CR6
	BGELR_CR6
	LXVD2X	(R8+R14), V0
	LXVD2X	(R4+R14), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	LXVD2X	(R8+R15), V0
	LXVD2X	(R4+R15), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	LXVD2X	(R8+R16), V0
	LXVD2X	(R4+R16), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	ADD	$64,R8		// bump up to next 64
	ADD	$64,R4
	BDNZ	loop64

	ISEL	CR0EQ, R11, R3, R3	// If no tail, return 1, otherwise R3 remains 0.
	BEQLR				// return if no tail.

	ADD	$-64, R9, R8
	ADD	$-64, R10, R4
	LXVD2X	(R8+R0), V0
	LXVD2X	(R4+R0), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	LXVD2X	(R8+R14), V0
	LXVD2X	(R4+R14), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	LXVD2X	(R8+R15), V0
	LXVD2X	(R4+R15), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	LXVD2X	(R8+R16), V0
	LXVD2X	(R4+R16), V1
	VCMPEQUBCC	V0, V1, V2
	ISEL	CR6LT, R11, R0, R3
	RET

check33_64:
	// Bytes 0-15
	LXVD2X	(R8+R0), V0
	LXVD2X	(R4+R0), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6
	ADD	$16, R8
	ADD	$16, R4

	// Bytes 16-31
	LXVD2X	(R8+R0), V0
	LXVD2X	(R4+R0), V1
	VCMPEQUBCC	V0, V1, V2
	BGELR_CR6

	// A little tricky, but point R4,R8 to &sx[len-32],
	// and reuse check17_32 to check the next 1-31 bytes (with some overlap)
	ADD	$-32, R9, R8
	ADD	$-32, R10, R4
	// Fallthrough

check17_32:
	LXVD2X	(R8+R0), V0
	LXVD2X	(R4+R0), V1
	VCMPEQUBCC	V0, V1, V2
	ISEL	CR6LT, R11, R0, R5

	// Load sX[len(sX)-16:len(sX)] and compare.
	ADD	$-16, R9
	ADD	$-16, R10
	LXVD2X	(R9+R0), V0
	LXVD2X	(R10+R0), V1
	VCMPEQUBCC	V0, V1, V2
	ISEL	CR6LT, R5, R0, R3
	RET

check0_16:
#ifdef GOPPC64_power10
	SLD	$56, R5, R7
	LXVL	R8, R7, V0
	LXVL	R4, R7, V1
	VCMPEQUDCC	V0, V1, V2
	ISEL	CR6LT, R11, R0, R3
	RET
#else
	CMP	R5, $8
	BLT	check0_7
	// Load sX[0:7] and compare.
	MOVD	(R8), R6
	MOVD	(R4), R7
	CMP	R6, R7
	ISEL	CR0EQ, R11, R0, R5
	// Load sX[len(sX)-8:len(sX)] and compare.
	MOVD	-8(R9), R6
	MOVD	-8(R10), R7
	CMP	R6, R7
	ISEL	CR0EQ, R5, R0, R3
	RET

check0_7:
	CMP	R5,$0
	MOVD	$1, R3
	BEQLR		// return if len == 0

	// Check < 8B loads with a single compare, but select the load address
	// such that it cannot cross a page boundary. Load a few bytes from the
	// lower address if that does not cross the lower page. Or, load a few
	// extra bytes from the higher addresses. And align those values
	// consistently in register as either address may have differing
	// alignment requirements.
	ANDCC	$PAGE_OFFSET, R8, R6	// &sX & PAGE_OFFSET
	ANDCC	$PAGE_OFFSET, R4, R9
	SUBC	R5, $8, R12		// 8-len
	SLD	$3, R12, R14		// (8-len)*8
	CMPU	R6, R12, CR1		// Enough bytes lower in the page to load lower?
	CMPU	R9, R12, CR0
	SUB	R12, R8, R6		// compute lower load address
	SUB	R12, R4, R9
	ISEL	CR1LT, R8, R6, R8	// R8 = R6 < 0 ? R8 (&s1) : R6 (&s1 - (8-len))
	ISEL	CR0LT, R4, R9, R4	// Similar for s2
	MOVD	(R8), R15
	MOVD	(R4), R16
	SLD	R14, R15, R7
	SLD	R14, R16, R17
	SRD	R14, R7, R7		// Clear the upper (8-len) bytes (with 2 shifts)
	SRD	R14, R17, R17
	SRD	R14, R15, R6		// Clear the lower (8-len) bytes
	SRD	R14, R16, R9
#ifdef GOARCH_ppc64le
	ISEL	CR1LT, R7, R6, R8      // Choose the correct len bytes to compare based on alignment
	ISEL	CR0LT, R17, R9, R4
#else
	ISEL	CR1LT, R6, R7, R8
	ISEL	CR0LT, R9, R17, R4
#endif
	CMP	R4, R8
	ISEL	CR0EQ, R11, R0, R3
	RET
#endif	// tail processing if !defined(GOPPC64_power10)
