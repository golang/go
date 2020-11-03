// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

TEXT ·Cas(SB),NOSPLIT,$0-13
	MOVW	ptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R5
	SYNC
try_cas:
	MOVW	R5, R3
	LL	(R1), R4	// R4 = *R1
	BNE	R2, R4, cas_fail
	SC	R3, (R1)	// *R1 = R3
	BEQ	R3, try_cas
	SYNC
	MOVB	R3, ret+12(FP)
	RET
cas_fail:
	MOVB	R0, ret+12(FP)
	RET

TEXT ·Store(SB),NOSPLIT,$0-8
	MOVW	ptr+0(FP), R1
	MOVW	val+4(FP), R2
	SYNC
	MOVW	R2, 0(R1)
	SYNC
	RET

TEXT ·Store8(SB),NOSPLIT,$0-5
	MOVW	ptr+0(FP), R1
	MOVB	val+4(FP), R2
	SYNC
	MOVB	R2, 0(R1)
	SYNC
	RET

TEXT ·Load(SB),NOSPLIT,$0-8
	MOVW	ptr+0(FP), R1
	SYNC
	MOVW	0(R1), R1
	SYNC
	MOVW	R1, ret+4(FP)
	RET

TEXT ·Load8(SB),NOSPLIT,$0-5
	MOVW	ptr+0(FP), R1
	SYNC
	MOVB	0(R1), R1
	SYNC
	MOVB	R1, ret+4(FP)
	RET

TEXT ·Xadd(SB),NOSPLIT,$0-12
	MOVW	ptr+0(FP), R2
	MOVW	delta+4(FP), R3
	SYNC
try_xadd:
	LL	(R2), R1	// R1 = *R2
	ADDU	R1, R3, R4
	MOVW	R4, R1
	SC	R4, (R2)	// *R2 = R4
	BEQ	R4, try_xadd
	SYNC
	MOVW	R1, ret+8(FP)
	RET

TEXT ·Xchg(SB),NOSPLIT,$0-12
	MOVW	ptr+0(FP), R2
	MOVW	new+4(FP), R5
	SYNC
try_xchg:
	MOVW	R5, R3
	LL	(R2), R1	// R1 = *R2
	SC	R3, (R2)	// *R2 = R3
	BEQ	R3, try_xchg
	SYNC
	MOVW	R1, ret+8(FP)
	RET

TEXT ·Casuintptr(SB),NOSPLIT,$0-13
	JMP	·Cas(SB)

TEXT ·CasRel(SB),NOSPLIT,$0-13
	JMP	·Cas(SB)

TEXT ·Loaduintptr(SB),NOSPLIT,$0-8
	JMP	·Load(SB)

TEXT ·Loaduint(SB),NOSPLIT,$0-8
	JMP	·Load(SB)

TEXT ·Loadp(SB),NOSPLIT,$-0-8
	JMP	·Load(SB)

TEXT ·Storeuintptr(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

TEXT ·Xadduintptr(SB),NOSPLIT,$0-12
	JMP	·Xadd(SB)

TEXT ·Loadint64(SB),NOSPLIT,$0-12
	JMP	·Load64(SB)

TEXT ·Xaddint64(SB),NOSPLIT,$0-20
	JMP	·Xadd64(SB)

TEXT ·Casp1(SB),NOSPLIT,$0-13
	JMP	·Cas(SB)

TEXT ·Xchguintptr(SB),NOSPLIT,$0-12
	JMP	·Xchg(SB)

TEXT ·StorepNoWB(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

TEXT ·StoreRel(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

TEXT ·StoreReluintptr(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

// void	Or8(byte volatile*, byte);
TEXT ·Or8(SB),NOSPLIT,$0-5
	MOVW	ptr+0(FP), R1
	MOVBU	val+4(FP), R2
	MOVW	$~3, R3	// Align ptr down to 4 bytes so we can use 32-bit load/store.
	AND	R1, R3
#ifdef GOARCH_mips
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R1
#endif
	AND	$3, R1, R4	// R4 = ((ptr & 3) * 8)
	SLL	$3, R4
	SLL	R4, R2, R2	// Shift val for aligned ptr. R2 = val << R4
	SYNC
try_or8:
	LL	(R3), R4	// R4 = *R3
	OR	R2, R4
	SC	R4, (R3)	// *R3 = R4
	BEQ	R4, try_or8
	SYNC
	RET

// void	And8(byte volatile*, byte);
TEXT ·And8(SB),NOSPLIT,$0-5
	MOVW	ptr+0(FP), R1
	MOVBU	val+4(FP), R2
	MOVW	$~3, R3
	AND	R1, R3
#ifdef GOARCH_mips
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R1
#endif
	AND	$3, R1, R4	// R4 = ((ptr & 3) * 8)
	SLL	$3, R4
	MOVW	$0xFF, R5
	SLL	R4, R2
	SLL	R4, R5
	NOR	R0, R5
	OR	R5, R2	// Shift val for aligned ptr. R2 = val << R4 | ^(0xFF << R4)
	SYNC
try_and8:
	LL	(R3), R4	// R4 = *R3
	AND	R2, R4
	SC	R4, (R3)	// *R3 = R4
	BEQ	R4, try_and8
	SYNC
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-8
	MOVW	ptr+0(FP), R1
	MOVW	val+4(FP), R2

	SYNC
	LL	(R1), R3
	OR	R2, R3
	SC	R3, (R1)
	BEQ	R3, -4(PC)
	SYNC
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-8
	MOVW	ptr+0(FP), R1
	MOVW	val+4(FP), R2

	SYNC
	LL	(R1), R3
	AND	R2, R3
	SC	R3, (R1)
	BEQ	R3, -4(PC)
	SYNC
	RET
