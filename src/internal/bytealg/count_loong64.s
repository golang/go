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

	// Implemented using 256-bit SMID instructions
lasxCountBody:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R8
	BEQ	R8, lsxCountBody
	XVMOVQ	R6, X0.B32

	// jump directly to lasx32 if length < 128
	MOVV	$128, R8
	BLT	R5, R8, lasx32
lasx128:
lasx128Loop:
	XVMOVQ	0(R4), X1
	XVMOVQ	32(R4), X2
	XVMOVQ	64(R4), X3
	XVMOVQ	96(R4), X4

	XVSEQB  X0, X1, X5
	XVSEQB  X0, X2, X6
	XVSEQB  X0, X3, X7
	XVSEQB  X0, X4, X8

	XVANDB  $1, X5, X5
	XVANDB  $1, X6, X6
	XVANDB  $1, X7, X7
	XVANDB  $1, X8, X8

	XVPCNTV	X5, X1
	XVPCNTV	X6, X2
	XVPCNTV	X7, X3
	XVPCNTV	X8, X4

	XVADDV	X2, X1
	XVADDV	X4, X3
	XVADDV	X3, X1

	XVMOVQ	X1.V[0], R9
	XVMOVQ	X1.V[1], R10
	XVMOVQ	X1.V[2], R11
	XVMOVQ	X1.V[3], R12

	ADDV	R9, R10
	ADDV	R11, R12
	ADDV	R10, R7
	ADDV	R12, R7

	ADDV	$-128, R5
	ADDV	$128, R4
	BGE	R5, R8, lasx128Loop

lasx32:
	// jump directly to lasx8 if length < 32
	MOVV	$32, R8
	BLT	R5, R8, lasx8
lasx32Loop:
	XVMOVQ	0(R4), X1
	XVSEQB  X0, X1, X2
	XVANDB  $1, X2, X2
	XVPCNTV	X2, X1
	XVMOVQ	X1.V[0], R9
	XVMOVQ	X1.V[1], R10
	XVMOVQ	X1.V[2], R11
	XVMOVQ	X1.V[3], R12
	ADDV	R9, R10
	ADDV	R11, R12
	ADDV	R10, R7
	ADDV	R12, R7
	ADDV	$-32, R5
	ADDV	$32, R4
	BGE	R5, R8, lasx32Loop
lasx8:
	// jump directly to tail if length < 8
	MOVV	$8, R8
	BLT	R5, R8, tail
lasx8Loop:
	MOVV	0(R4), R9
	VMOVQ	R9, V1.V[0]
	VSEQB	V0, V1, V2
	VANDB	$1, V2, V2
	VPCNTV	V2, V1

	VMOVQ	V1.V[0], R9
	ADDV	R9, R7
	ADDV	$-8, R5
	ADDV	$8, R4
	BGE	R5, R8, lasx8Loop
	JMP	tail

	// Implemented using 128-bit SMID instructions
lsxCountBody:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R8
	BEQ	R8, genericCountBody
	VMOVQ	R6, V0.B16

	// jump directly to lsx16 if length < 64
	MOVV	$64, R8
	BLT	R5, R8, lsx16
lsx64:
lsx64Loop:
	VMOVQ	0(R4),  V1
	VMOVQ	16(R4), V2
	VMOVQ	32(R4), V3
	VMOVQ	48(R4), V4

	VSEQB  V0, V1, V5
	VSEQB  V0, V2, V6
	VSEQB  V0, V3, V7
	VSEQB  V0, V4, V8

	VANDB  $1, V5, V5
	VANDB  $1, V6, V6
	VANDB  $1, V7, V7
	VANDB  $1, V8, V8

	VPCNTV	V5, V1
	VPCNTV	V6, V2
	VPCNTV	V7, V3
	VPCNTV	V8, V4

	VADDV	V2, V1
	VADDV	V4, V3
	VADDV	V3, V1

	VMOVQ	V1.V[0], R9
	VMOVQ	V1.V[1], R10
	ADDV	R9, R7
	ADDV	R10, R7

	ADDV	$-64, R5
	ADDV	$64, R4
	BGE	R5, R8, lsx64Loop

lsx16:
	// jump directly to lsx8 if length < 16
	MOVV	$16, R8
	BLT	R5, R8, lsx8
lsx16Loop:
	VMOVQ	0(R4), V1
	VSEQB	V0, V1, V2
	VANDB  $1, V2, V2
	VPCNTV	V2, V1
	VMOVQ	V1.V[0], R9
	VMOVQ	V1.V[1], R10
	ADDV	R9, R7
	ADDV	R10, R7
	ADDV	$-16, R5
	ADDV	$16, R4
	BGE	R5, R8, lsx16Loop
lsx8:
	// jump directly to tail if length < 8
	MOVV	$8, R8
	BLT	R5, R8, tail
lsx8Loop:
	MOVV	0(R4), R9
	VMOVQ	R9, V1.V[0]
	VSEQB	V0, V1, V2
	VANDB	$1, V2, V2
	VPCNTV	V2, V1

	VMOVQ	V1.V[0], R9
	ADDV	R9, R7
	ADDV	$-8, R5
	ADDV	$8, R4
	BGE	R5, R8, lsx8Loop
	JMP	tail

	// Implemented using general instructions
genericCountBody:
	MOVV	$4, R8
	MOVV	$1, R9
genericLoop:
	BLT	R5, R8, tail
	ADDV	$-4, R5
	MOVWU	(R4)(R5), R10
	BSTRPICKW	$7, R10, $0, R11
	BSTRPICKW	$15, R10, $8, R12
	XOR	R6, R11
	XOR	R6, R12
	MASKNEZ	R11, R9, R13
	MASKNEZ	R12, R9, R14
	ADDV	R13, R7
	ADDV	R14, R7
	BSTRPICKW	$23, R10, $16, R11
	BSTRPICKW	$31, R10, $24, R12
	XOR	R6, R11
	XOR	R6, R12
	MASKNEZ	R11, R9, R13
	MASKNEZ	R12, R9, R14
	ADDV	R13, R7
	ADDV	R14, R7
	JMP	genericLoop

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
