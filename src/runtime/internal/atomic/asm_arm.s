// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
//
// To implement runtime∕internal∕atomic·cas in sys_$GOOS_arm.s
// using the native instructions, use:
//
//	TEXT runtime∕internal∕atomic·cas(SB),NOSPLIT,$0
//		B	runtime∕internal∕atomic·armcas(SB)
//
TEXT runtime∕internal∕atomic·armcas(SB),NOSPLIT,$0-13
	MOVW	ptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casl:
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail

	MOVB	runtime·goarm(SB), R8
	CMP	$7, R8
	BLT	2(PC)
	DMB	MB_ISHST

	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casl
	MOVW	$1, R0

	CMP	$7, R8
	BLT	2(PC)
	DMB	MB_ISH

	MOVB	R0, ret+12(FP)
	RET
casfail:
	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

// stubs

TEXT runtime∕internal∕atomic·Loadp(SB),NOSPLIT|NOFRAME,$0-8
	B runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·LoadAcq(SB),NOSPLIT|NOFRAME,$0-8
	B runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Casuintptr(SB),NOSPLIT,$0-13
	B	runtime∕internal∕atomic·Cas(SB)

TEXT runtime∕internal∕atomic·Casp1(SB),NOSPLIT,$0-13
	B	runtime∕internal∕atomic·Cas(SB)

TEXT runtime∕internal∕atomic·CasRel(SB),NOSPLIT,$0-13
	B	runtime∕internal∕atomic·Cas(SB)

TEXT runtime∕internal∕atomic·Loaduintptr(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Loaduint(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Storeuintptr(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Store(SB)

TEXT runtime∕internal∕atomic·StorepNoWB(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Store(SB)

TEXT runtime∕internal∕atomic·StoreRel(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Store(SB)

TEXT runtime∕internal∕atomic·Xadduintptr(SB),NOSPLIT,$0-12
	B	runtime∕internal∕atomic·Xadd(SB)

TEXT runtime∕internal∕atomic·Loadint64(SB),NOSPLIT,$0-12
	B	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Xaddint64(SB),NOSPLIT,$0-20
	B	runtime∕internal∕atomic·Xadd64(SB)

// 64-bit atomics
// The native ARM implementations use LDREXD/STREXD, which are
// available on ARMv6k or later. We use them only on ARMv7.
// On older ARM, we use Go implementations which simulate 64-bit
// atomics with locks.

TEXT	armCas64<>(SB),NOSPLIT,$0-21
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)	// crash. AND.S above left only low 3 bits in R2.
	MOVW	old_lo+4(FP), R2
	MOVW	old_hi+8(FP), R3
	MOVW	new_lo+12(FP), R4
	MOVW	new_hi+16(FP), R5
cas64loop:
	LDREXD	(R1), R6	// loads R6 and R7
	CMP	R2, R6
	BNE	cas64fail
	CMP	R3, R7
	BNE	cas64fail

	DMB	MB_ISHST

	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	cas64loop
	MOVW	$1, R0

	DMB	MB_ISH

	MOVBU	R0, swapped+20(FP)
	RET
cas64fail:
	MOVW	$0, R0
	MOVBU	R0, swapped+20(FP)
	RET

TEXT	armXadd64<>(SB),NOSPLIT,$0-20
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)	// crash. AND.S above left only low 3 bits in R2.
	MOVW	delta_lo+4(FP), R2
	MOVW	delta_hi+8(FP), R3

add64loop:
	LDREXD	(R1), R4	// loads R4 and R5
	ADD.S	R2, R4
	ADC	R3, R5

	DMB	MB_ISHST

	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	add64loop

	DMB	MB_ISH

	MOVW	R4, new_lo+12(FP)
	MOVW	R5, new_hi+16(FP)
	RET

TEXT	armXchg64<>(SB),NOSPLIT,$0-20
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)	// crash. AND.S above left only low 3 bits in R2.
	MOVW	new_lo+4(FP), R2
	MOVW	new_hi+8(FP), R3

swap64loop:
	LDREXD	(R1), R4	// loads R4 and R5

	DMB	MB_ISHST

	STREXD	R2, (R1), R0	// stores R2 and R3
	CMP	$0, R0
	BNE	swap64loop

	DMB	MB_ISH

	MOVW	R4, old_lo+12(FP)
	MOVW	R5, old_hi+16(FP)
	RET

TEXT	armLoad64<>(SB),NOSPLIT,$0-12
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)	// crash. AND.S above left only low 3 bits in R2.

	LDREXD	(R1), R2	// loads R2 and R3
	DMB	MB_ISH

	MOVW	R2, val_lo+4(FP)
	MOVW	R3, val_hi+8(FP)
	RET

TEXT	armStore64<>(SB),NOSPLIT,$0-12
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)	// crash. AND.S above left only low 3 bits in R2.
	MOVW	val_lo+4(FP), R2
	MOVW	val_hi+8(FP), R3

store64loop:
	LDREXD	(R1), R4	// loads R4 and R5

	DMB	MB_ISHST

	STREXD	R2, (R1), R0	// stores R2 and R3
	CMP	$0, R0
	BNE	store64loop

	DMB	MB_ISH
	RET

TEXT	·Cas64(SB),NOSPLIT,$0-21
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	armCas64<>(SB)
	JMP	·goCas64(SB)

TEXT	·Xadd64(SB),NOSPLIT,$0-20
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	armXadd64<>(SB)
	JMP	·goXadd64(SB)

TEXT	·Xchg64(SB),NOSPLIT,$0-20
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	armXchg64<>(SB)
	JMP	·goXchg64(SB)

TEXT	·Load64(SB),NOSPLIT,$0-12
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	armLoad64<>(SB)
	JMP	·goLoad64(SB)

TEXT	·Store64(SB),NOSPLIT,$0-12
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	armStore64<>(SB)
	JMP	·goStore64(SB)
