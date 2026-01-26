// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips64 || mips64le

#include "go_asm.h"
#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-16
	MOVV	ptr+0(FP), R1
	MOVV	n+8(FP), R2
	ADDV	R1, R2, R4

	// if less than 16 bytes or no MSA, do words check
	SGTU	$16, R2, R3
	BNE	R3, no_msa
	MOVBU	internal∕cpu·MIPS64X+const_offsetMIPS64XHasMSA(SB), R3
	BEQ	R3, R0, no_msa

	VMOVB	$0, W0

	SGTU	$128, R2, R3
	BEQ	R3, msa_large

	AND	$15, R2, R5
	XOR	R2, R5, R6
	ADDVU	R1, R6

msa_small:
	VMOVB	W0, (R1)
	ADDVU	$16, R1
	SGTU	R6, R1, R3
	BNE	R3, R0, msa_small
	BEQ	R5, R0, done
	VMOVB	W0, -16(R4)
	JMP	done

msa_large:
	AND	$127, R2, R5
	XOR	R2, R5, R6
	ADDVU	R1, R6

msa_large_loop:
	VMOVB	W0, (R1)
	VMOVB	W0, 16(R1)
	VMOVB	W0, 32(R1)
	VMOVB	W0, 48(R1)
	VMOVB	W0, 64(R1)
	VMOVB	W0, 80(R1)
	VMOVB	W0, 96(R1)
	VMOVB	W0, 112(R1)

	ADDVU	$128, R1
	SGTU	R6, R1, R3
	BNE	R3, R0, msa_large_loop
	BEQ	R5, R0, done
	VMOVB	W0, -128(R4)
	VMOVB	W0, -112(R4)
	VMOVB	W0, -96(R4)
	VMOVB	W0, -80(R4)
	VMOVB	W0, -64(R4)
	VMOVB	W0, -48(R4)
	VMOVB	W0, -32(R4)
	VMOVB	W0, -16(R4)
	JMP	done

no_msa:
	// if less than 8 bytes, do one byte at a time
	SGTU	$8, R2, R3
	BNE	R3, check4

	// Check alignment
	AND	$7, R1, R3
	BEQ	R3, aligned

	// Zero one byte at a time until we reach 8 byte alignment.
	MOVV	$8, R5
	SUBV	R3, R5, R3
	SUBV	R3, R2, R2
align:
	SUBV	$1, R3
	MOVB	R0, (R1)
	ADDV	$1, R1
	BNE	R3, align

aligned:
	SGTU	$8, R2, R3
	BNE	R3, check4
	SGTU	$16, R2, R3
	BNE	R3, zero8
	SGTU	$32, R2, R3
	BNE	R3, zero16
	SGTU	$64, R2, R3
	BNE	R3, zero32
loop64:
	MOVV	R0, (R1)
	MOVV	R0, 8(R1)
	MOVV	R0, 16(R1)
	MOVV	R0, 24(R1)
	MOVV	R0, 32(R1)
	MOVV	R0, 40(R1)
	MOVV	R0, 48(R1)
	MOVV	R0, 56(R1)
	ADDV	$64, R1
	SUBV	$64, R2
	SGTU	$64, R2, R3
	BEQ	R0, R3, loop64
	BEQ	R2, done

check32:
	SGTU	$32, R2, R3
	BNE	R3, check16
zero32:
	MOVV	R0, (R1)
	MOVV	R0, 8(R1)
	MOVV	R0, 16(R1)
	MOVV	R0, 24(R1)
	ADDV	$32, R1
	SUBV	$32, R2
	BEQ	R2, done

check16:
	SGTU	$16, R2, R3
	BNE	R3, check8
zero16:
	MOVV	R0, (R1)
	MOVV	R0, 8(R1)
	ADDV	$16, R1
	SUBV	$16, R2
	BEQ	R2, done

check8:
	SGTU	$8, R2, R3
	BNE	R3, check4
zero8:
	MOVV	R0, (R1)
	ADDV	$8, R1
	SUBV	$8, R2
	BEQ	R2, done

check4:
	SGTU	$4, R2, R3
	BNE	R3, loop1
zero4:
	MOVB	R0, (R1)
	MOVB	R0, 1(R1)
	MOVB	R0, 2(R1)
	MOVB	R0, 3(R1)
	ADDV	$4, R1
	SUBV	$4, R2

loop1:
	BEQ	R1, R4, done
	MOVB	R0, (R1)
	ADDV	$1, R1
	JMP	loop1
done:
	RET

