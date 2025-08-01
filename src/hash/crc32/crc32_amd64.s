// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "go_asm.h"

// castagnoliSSE42 updates the (non-inverted) crc with the given buffer.
//
// func castagnoliSSE42(crc uint32, p []byte) uint32
TEXT ·castagnoliSSE42(SB),NOSPLIT,$0
	MOVL crc+0(FP), AX  // CRC value
	MOVQ p+8(FP), SI  // data pointer
	MOVQ p_len+16(FP), CX  // len(p)

	// If there are fewer than 8 bytes to process, skip alignment.
	CMPQ CX, $8
	JL less_than_8

	MOVQ SI, BX
	ANDQ $7, BX
	JZ aligned

	// Process the first few bytes to 8-byte align the input.

	// BX = 8 - BX. We need to process this many bytes to align.
	SUBQ $1, BX
	XORQ $7, BX

	BTQ $0, BX
	JNC align_2

	CRC32B (SI), AX
	DECQ CX
	INCQ SI

align_2:
	BTQ $1, BX
	JNC align_4

	CRC32W (SI), AX

	SUBQ $2, CX
	ADDQ $2, SI

align_4:
	BTQ $2, BX
	JNC aligned

	CRC32L (SI), AX

	SUBQ $4, CX
	ADDQ $4, SI

aligned:
	// The input is now 8-byte aligned and we can process 8-byte chunks.
	CMPQ CX, $8
	JL less_than_8

	CRC32Q (SI), AX
	ADDQ $8, SI
	SUBQ $8, CX
	JMP aligned

less_than_8:
	// We may have some bytes left over; process 4 bytes, then 2, then 1.
	BTQ $2, CX
	JNC less_than_4

	CRC32L (SI), AX
	ADDQ $4, SI

less_than_4:
	BTQ $1, CX
	JNC less_than_2

	CRC32W (SI), AX
	ADDQ $2, SI

less_than_2:
	BTQ $0, CX
	JNC done

	CRC32B (SI), AX

done:
	MOVL AX, ret+32(FP)
	RET

// castagnoliSSE42Triple updates three (non-inverted) crcs with (24*rounds)
// bytes from each buffer.
//
// func castagnoliSSE42Triple(
//     crc1, crc2, crc3 uint32,
//     a, b, c []byte,
//     rounds uint32,
// ) (retA uint32, retB uint32, retC uint32)
TEXT ·castagnoliSSE42Triple(SB),NOSPLIT,$0
	MOVL crcA+0(FP), AX
	MOVL crcB+4(FP), CX
	MOVL crcC+8(FP), DX

	MOVQ a+16(FP), R8   // data pointer
	MOVQ b+40(FP), R9   // data pointer
	MOVQ c+64(FP), R10  // data pointer

	MOVL rounds+88(FP), R11

loop:
	CRC32Q (R8), AX
	CRC32Q (R9), CX
	CRC32Q (R10), DX

	CRC32Q 8(R8), AX
	CRC32Q 8(R9), CX
	CRC32Q 8(R10), DX

	CRC32Q 16(R8), AX
	CRC32Q 16(R9), CX
	CRC32Q 16(R10), DX

	ADDQ $24, R8
	ADDQ $24, R9
	ADDQ $24, R10

	DECQ R11
	JNZ loop

	MOVL AX, retA+96(FP)
	MOVL CX, retB+100(FP)
	MOVL DX, retC+104(FP)
	RET

// CRC32 polynomial data
//
// These constants are lifted from the
// Linux kernel, since they avoid the costly
// PSHUFB 16 byte reversal proposed in the
// original Intel paper.
// Splatted so it can be loaded with a single VMOVDQU64
DATA r2r1<>+0(SB)/8, $0x154442bd4
DATA r2r1<>+8(SB)/8, $0x1c6e41596
DATA r2r1<>+16(SB)/8, $0x154442bd4
DATA r2r1<>+24(SB)/8, $0x1c6e41596
DATA r2r1<>+32(SB)/8, $0x154442bd4
DATA r2r1<>+40(SB)/8, $0x1c6e41596
DATA r2r1<>+48(SB)/8, $0x154442bd4
DATA r2r1<>+56(SB)/8, $0x1c6e41596

DATA r4r3<>+0(SB)/8, $0x1751997d0
DATA r4r3<>+8(SB)/8, $0x0ccaa009e
DATA rupoly<>+0(SB)/8, $0x1db710641
DATA rupoly<>+8(SB)/8, $0x1f7011641
DATA r5<>+0(SB)/8, $0x163cd6124

GLOBL r2r1<>(SB), RODATA, $64
GLOBL r4r3<>(SB),RODATA,$16
GLOBL rupoly<>(SB),RODATA,$16
GLOBL r5<>(SB),RODATA,$8

// Based on https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf
// len(p) must be at least 64, and must be a multiple of 16.

// func ieeeCLMUL(crc uint32, p []byte) uint32
TEXT ·ieeeCLMUL(SB),NOSPLIT,$0
	MOVL   crc+0(FP), X0             // Initial CRC value
	MOVQ   p+8(FP), SI  	         // data pointer
	MOVQ   p_len+16(FP), CX          // len(p)

	// Check feature support and length to be >= 1024 bytes.
	CMPB internal∕cpu·X86+const_offsetX86HasAVX512VPCLMULQDQL(SB), $1
	JNE  useSSE42
	CMPQ CX, $1024
	JL   useSSE42

	// Use AVX512. Zero upper and Z10 and load initial CRC into lower part of Z10.
	VPXORQ    Z10, Z10, Z10
	VMOVAPS   X0, X10
	VMOVDQU64 (SI), Z1
	VPXORQ    Z10, Z1, Z1 // Merge initial CRC value into Z1
	ADDQ      $64, SI    // buf+=64
	SUBQ      $64, CX    // len-=64

	VMOVDQU64 r2r1<>+0(SB), Z0

loopback64Avx512:
	VMOVDQU64  (SI), Z11          // Load next
	VPCLMULQDQ $0x11, Z0, Z1, Z5
	VPCLMULQDQ $0, Z0, Z1, Z1
	VPTERNLOGD $0x96, Z11, Z5, Z1 // Combine results with xor into Z1

	ADDQ $0x40, DI
	ADDQ $64, SI    // buf+=64
	SUBQ $64, CX    // len-=64
	CMPQ CX, $64    // Less than 64 bytes left?
	JGE  loopback64Avx512

	// Unfold result into XMM1-XMM4 to match SSE4 code.
	VEXTRACTF32X4 $1, Z1, X2 // X2: Second 128-bit lane
	VEXTRACTF32X4 $2, Z1, X3 // X3: Third 128-bit lane
	VEXTRACTF32X4 $3, Z1, X4 // X4: Fourth 128-bit lane
	VZEROUPPER
	JMP remain64

	PCALIGN $16
useSSE42:
	MOVOU  (SI), X1
	MOVOU  16(SI), X2
	MOVOU  32(SI), X3
	MOVOU  48(SI), X4
	PXOR   X0, X1
	ADDQ   $64, SI                  // buf+=64
	SUBQ   $64, CX                  // len-=64
	CMPQ   CX, $64                  // Less than 64 bytes left
	JB     remain64

	MOVOA  r2r1<>+0(SB), X0
loopback64:
	MOVOA  X1, X5
	MOVOA  X2, X6
	MOVOA  X3, X7
	MOVOA  X4, X8

	PCLMULQDQ $0, X0, X1
	PCLMULQDQ $0, X0, X2
	PCLMULQDQ $0, X0, X3
	PCLMULQDQ $0, X0, X4

	/* Load next early */
	MOVOU    (SI), X11
	MOVOU    16(SI), X12
	MOVOU    32(SI), X13
	MOVOU    48(SI), X14

	PCLMULQDQ $0x11, X0, X5
	PCLMULQDQ $0x11, X0, X6
	PCLMULQDQ $0x11, X0, X7
	PCLMULQDQ $0x11, X0, X8

	PXOR     X5, X1
	PXOR     X6, X2
	PXOR     X7, X3
	PXOR     X8, X4

	PXOR     X11, X1
	PXOR     X12, X2
	PXOR     X13, X3
	PXOR     X14, X4

	ADDQ    $0x40, DI
	ADDQ    $64, SI      // buf+=64
	SUBQ    $64, CX      // len-=64
	CMPQ    CX, $64      // Less than 64 bytes left?
	JGE     loopback64

	PCALIGN $16
	/* Fold result into a single register (X1) */
remain64:
	MOVOA       r4r3<>+0(SB), X0

	MOVOA       X1, X5
	PCLMULQDQ   $0, X0, X1
	PCLMULQDQ   $0x11, X0, X5
	PXOR        X5, X1
	PXOR        X2, X1

	MOVOA       X1, X5
	PCLMULQDQ   $0, X0, X1
	PCLMULQDQ   $0x11, X0, X5
	PXOR        X5, X1
	PXOR        X3, X1

	MOVOA       X1, X5
	PCLMULQDQ   $0, X0, X1
	PCLMULQDQ   $0x11, X0, X5
	PXOR        X5, X1
	PXOR        X4, X1

	/* If there is less than 16 bytes left we are done */
	CMPQ        CX, $16
	JB          finish

	/* Encode 16 bytes */
remain16:
	MOVOU       (SI), X10
	MOVOA       X1, X5
	PCLMULQDQ   $0, X0, X1
	PCLMULQDQ   $0x11, X0, X5
	PXOR        X5, X1
	PXOR        X10, X1
	SUBQ        $16, CX
	ADDQ        $16, SI
	CMPQ        CX, $16
	JGE         remain16

finish:
	/* Fold final result into 32 bits and return it */
	PCMPEQB     X3, X3
	PCLMULQDQ   $1, X1, X0
	PSRLDQ      $8, X1
	PXOR        X0, X1

	MOVOA       X1, X2
	MOVQ        r5<>+0(SB), X0

	/* Creates 32 bit mask. Note that we don't care about upper half. */
	PSRLQ       $32, X3

	PSRLDQ      $4, X2
	PAND        X3, X1
	PCLMULQDQ   $0, X0, X1
	PXOR        X2, X1

	MOVOA       rupoly<>+0(SB), X0

	MOVOA       X1, X2
	PAND        X3, X1
	PCLMULQDQ   $0x10, X0, X1
	PAND        X3, X1
	PCLMULQDQ   $0, X0, X1
	PXOR        X2, X1

	PEXTRD	$1, X1, AX
	MOVL        AX, ret+32(FP)

	RET
