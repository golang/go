// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count<ABIInternal>(SB),NOSPLIT,$0-40
	// R4 = b_base
	// R5 = b_len
	// R6 = b_cap (unused)
	// R7 = byte to count
	AND	$0xff, R7, R6
	JMP	countbody<>(SB)

TEXT ·CountString<ABIInternal>(SB),NOSPLIT,$0-32
	// R4 = s_base
	// R5 = s_len
	// R6 = byte to count
	AND	$0xff, R6
	JMP	countbody<>(SB)

// input:
//   R4 = s_base
//   R5 = s_len
//   R6 = byte to count
TEXT countbody<>(SB),NOSPLIT,$0
	MOVV	R0, R7	// count

	// short path to handle 0-byte case
	BEQ	R5, done

	// jump directly to tail length < 8
	MOVV	$8, R8
	BLT	R5, R8, tail

	// Implemented using 256-bit SIMD instructions
lasxCountBody:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R8
	BEQ	R8, lsxCountBody
	XVMOVQ	R6, X0.B32
	XVXORV  X9, X9, X9              // byte accumulator, must be zeroed

	// jump directly to lasx32 if length < 128
	MOVV	$128, R8
	BLT	R5, R8, lasx32

lasx128Outer:
	MOVV	$7, R14
	SRLV	$7, R5, R15		// R15 = R5 / 128
	BGE	R14, R15, lasx128Loop
	MOVV	R14, R15		// cap at 7 iterations

lasx128Loop:
	XVMOVQ	0(R4), X1
	XVMOVQ	32(R4), X2
	XVMOVQ	64(R4), X3
	XVMOVQ	96(R4), X4

	XVSEQB  X0, X1, X5
	XVSEQB  X0, X2, X6
	XVSEQB  X0, X3, X7
	XVSEQB  X0, X4, X8

	XVSUBB	X5, X9, X9
	XVSUBB	X6, X9, X9
	XVSUBB	X7, X9, X9
	XVSUBB	X8, X9, X9

	ADDV	$-128, R5
	ADDV	$128, R4

	ADDV	$-1, R15
	BNE	R15, lasx128Loop

	// flush byte accumulator X9 into R7:
	// sum the 8 bytes of each 64-bit lane via (x * 0x0101..) >> 56
	MOVV	$0x0101010101010101, R17
	XVMOVQ	X9.V[0], R9
	XVMOVQ	X9.V[1], R10
	XVMOVQ	X9.V[2], R11
	XVMOVQ	X9.V[3], R12
	MULV	R17, R9
	MULV	R17, R10
	MULV	R17, R11
	MULV	R17, R12
	SRLV	$56, R9
	SRLV	$56, R10
	SRLV	$56, R11
	SRLV	$56, R12
	ADDV	R9, R10
	ADDV	R11, R12
	ADDV	R10, R7
	ADDV	R12, R7
	XVXORV	X9, X9, X9
	BGE	R5, R8, lasx128Outer

lasx32:
	// jump directly to scalar8 if length < 32
	MOVV	$32, R8
	BLT	R5, R8, scalar8
lasx32Loop:
	XVMOVQ	0(R4), X1
	XVSEQB  X0, X1, X2
	XVSUBB  X2, X9, X9
	ADDV	$-32, R5
	ADDV	$32, R4
	BGE	R5, R8, lasx32Loop

	// final flush of X9 (<= 3 iterations accumulated above)
	MOVV    $0x0101010101010101, R17
	XVMOVQ  X9.V[0], R9
	XVMOVQ  X9.V[1], R10
	XVMOVQ  X9.V[2], R11
	XVMOVQ  X9.V[3], R12
	MULV    R17, R9
	MULV    R17, R10
	MULV    R17, R11
	MULV    R17, R12
	SRLV    $56, R9
	SRLV    $56, R10
	SRLV    $56, R11
	SRLV    $56, R12
	ADDV    R9, R10
	ADDV    R11, R12
	ADDV    R10, R7
	ADDV    R12, R7
	JMP	scalar8

	// Implemented using 128-bit SIMD instructions
lsxCountBody:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R8
	BEQ	R8, genericCountBody
	VMOVQ	R6, V0.B16
	VXORV   V9, V9, V9

	// jump directly to lsx16 if length < 64
	MOVV	$64, R8
	BLT	R5, R8, lsx16

lsx64Outer:
	MOVV	$7, R14
	SRLV	$6, R5, R15		// R15 = R5 / 64
	BGE	R14, R15, lsx64Loop
	MOVV	R14, R15

lsx64Loop:
	VMOVQ	0(R4),  V1
	VMOVQ	16(R4), V2
	VMOVQ	32(R4), V3
	VMOVQ	48(R4), V4

	VSEQB  V0, V1, V5
	VSEQB  V0, V2, V6
	VSEQB  V0, V3, V7
	VSEQB  V0, V4, V8

	VSUBB	V5, V9, V9
	VSUBB	V6, V9, V9
	VSUBB	V7, V9, V9
	VSUBB	V8, V9, V9

	ADDV	$-64, R5
	ADDV	$64, R4
	ADDV	$-1, R15
	BNE	R15, lsx64Loop

	MOVV	$0x0101010101010101, R17
	VMOVQ	V9.V[0], R9
	VMOVQ	V9.V[1], R10
	MULV	R17, R9
	MULV	R17, R10
	SRLV	$56, R9
	SRLV	$56, R10
	ADDV	R9, R7
	ADDV	R10, R7
	VXORV	V9, V9, V9
	BGE	R5, R8, lsx64Outer

lsx16:
	// jump directly to scalar8 if length < 16
	MOVV	$16, R8
	BLT	R5, R8, scalar8
lsx16Loop:
	VMOVQ	0(R4), V1
	VSEQB	V0, V1, V2
	VSUBB	V2, V9, V9
	ADDV	$-16, R5
	ADDV	$16, R4
	BGE	R5, R8, lsx16Loop

	// final flush of V9
	MOVV    $0x0101010101010101, R17
	VMOVQ   V9.V[0], R9
	VMOVQ   V9.V[1], R10
	MULV    R17, R9
	MULV    R17, R10
	SRLV    $56, R9
	SRLV    $56, R10
	ADDV    R9, R7
	ADDV    R10, R7
	JMP	scalar8

	// Implemented using general instructions
genericCountBody:
	MOVV    $4, R8
	MOVV    $1, R9
genericLoop:
	BLT     R5, R8, tail
	ADDV    $-4, R5
	MOVWU   (R4)(R5), R10
	BSTRPICKW       $7, R10, $0, R11
	BSTRPICKW       $15, R10, $8, R12
	XOR     R6, R11
	XOR     R6, R12
	MASKNEZ R11, R9, R13
	MASKNEZ R12, R9, R14
	ADDV    R13, R7
	ADDV    R14, R7
	BSTRPICKW       $23, R10, $16, R11
	BSTRPICKW       $31, R10, $24, R12
	XOR     R6, R11
	XOR     R6, R12
	MASKNEZ R11, R9, R13
	MASKNEZ R12, R9, R14
	ADDV    R13, R7
	ADDV    R14, R7
	JMP     genericLoop

	// Scalar 8-byte loop for the <32/<16 byte remainders of the
	// LASX/LSX paths. Counts bytes equal to R6 using an exact SWAR
	// zero-byte test:
	//   t       = x ^ broadcast(R6)
	//   nonzero = (((t & 0x7f7f..) + 0x7f7f..) | t) & 0x8080..
	//   match   = nonzero ^ 0x8080..   (0x80 in each matching byte)
	// This form has no inter-byte borrows, so adjacent matches are
	// counted correctly (unlike the (x-0x01..) & ~x & 0x80.. form).
scalar8:
	MOVV	$8, R8
	BLT	R5, R8, tail
	MOVV	$0x0101010101010101, R17
	MOVV	$0x7f7f7f7f7f7f7f7f, R19
	MOVV	$0x8080808080808080, R18
	MULV	R17, R6, R16		// R16 = R6 broadcast to 8 bytes
scalar8Loop:
	MOVV	0(R4), R10
	XOR	R16, R10		// t
	AND	R19, R10, R11		// t & 0x7f7f..
	ADDV	R19, R11		// + 0x7f7f.. (no carry out)
	OR	R10, R11
	AND	R18, R11		// 0x80 in each nonzero byte
	XOR	R18, R11		// 0x80 in each matching byte
	SRLV	$7, R11			// 0x01 per match
	MULV	R17, R11
	SRLV	$56, R11		// R11 = match count in word
	ADDV	R11, R7
	ADDV	$-8, R5
	ADDV	$8, R4
	BGE	R5, R8, scalar8Loop
	JMP	tail

	// Work with tail shorter than 8 bytes
tail:
	BEQ	R5, done
	ADDV	$-1, R5
	MOVBU   (R4)(R5), R8
	BNE	R6, R8, tail
	ADDV	$1, R7
	JMP	tail
done:
	MOVV	R7, R4
	RET
