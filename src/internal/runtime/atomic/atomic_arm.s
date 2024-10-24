// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"

// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
//
// To implement ·cas in sys_$GOOS_arm.s
// using the native instructions, use:
//
//	TEXT ·cas(SB),NOSPLIT,$0
//		B	·armcas(SB)
//
TEXT ·armcas(SB),NOSPLIT,$0-13
	MOVW	ptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casl:
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$0, R11
	BEQ	2(PC)
#endif
	DMB	MB_ISHST

	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casl
	MOVW	$1, R0

#ifndef GOARM_7
	CMP	$0, R11
	BEQ	2(PC)
#endif
	DMB	MB_ISH

	MOVB	R0, ret+12(FP)
	RET
casfail:
	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

// stubs

TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$0-8
	B	·Load(SB)

TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-8
	B	·Load(SB)

TEXT ·LoadAcquintptr(SB),NOSPLIT|NOFRAME,$0-8
	B 	·Load(SB)

TEXT ·Casint32(SB),NOSPLIT,$0-13
	B	·Cas(SB)

TEXT ·Casint64(SB),NOSPLIT,$-4-21
	B	·Cas64(SB)

TEXT ·Casuintptr(SB),NOSPLIT,$0-13
	B	·Cas(SB)

TEXT ·Casp1(SB),NOSPLIT,$0-13
	B	·Cas(SB)

TEXT ·CasRel(SB),NOSPLIT,$0-13
	B	·Cas(SB)

TEXT ·Loadint32(SB),NOSPLIT,$0-8
	B	·Load(SB)

TEXT ·Loadint64(SB),NOSPLIT,$-4-12
	B	·Load64(SB)

TEXT ·Loaduintptr(SB),NOSPLIT,$0-8
	B	·Load(SB)

TEXT ·Loaduint(SB),NOSPLIT,$0-8
	B	·Load(SB)

TEXT ·Storeint32(SB),NOSPLIT,$0-8
	B	·Store(SB)

TEXT ·Storeint64(SB),NOSPLIT,$0-12
	B	·Store64(SB)

TEXT ·Storeuintptr(SB),NOSPLIT,$0-8
	B	·Store(SB)

TEXT ·StorepNoWB(SB),NOSPLIT,$0-8
	B	·Store(SB)

TEXT ·StoreRel(SB),NOSPLIT,$0-8
	B	·Store(SB)

TEXT ·StoreReluintptr(SB),NOSPLIT,$0-8
	B	·Store(SB)

TEXT ·Xaddint32(SB),NOSPLIT,$0-12
	B	·Xadd(SB)

TEXT ·Xaddint64(SB),NOSPLIT,$-4-20
	B	·Xadd64(SB)

TEXT ·Xadduintptr(SB),NOSPLIT,$0-12
	B	·Xadd(SB)

TEXT ·Xchgint32(SB),NOSPLIT,$0-12
	B	·Xchg(SB)

TEXT ·Xchgint64(SB),NOSPLIT,$-4-20
	B	·Xchg64(SB)

// 64-bit atomics
// The native ARM implementations use LDREXD/STREXD, which are
// available on ARMv6k or later. We use them only on ARMv7.
// On older ARM, we use Go implementations which simulate 64-bit
// atomics with locks.
TEXT armCas64<>(SB),NOSPLIT,$0-21
	// addr is already in R1
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

TEXT armXadd64<>(SB),NOSPLIT,$0-20
	// addr is already in R1
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

TEXT armXchg64<>(SB),NOSPLIT,$0-20
	// addr is already in R1
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

TEXT armLoad64<>(SB),NOSPLIT,$0-12
	// addr is already in R1

	LDREXD	(R1), R2	// loads R2 and R3
	DMB	MB_ISH

	MOVW	R2, val_lo+4(FP)
	MOVW	R3, val_hi+8(FP)
	RET

TEXT armStore64<>(SB),NOSPLIT,$0-12
	// addr is already in R1
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

TEXT armAnd8<>(SB),NOSPLIT,$0-5
	// addr is already in R1
	MOVB	v+4(FP), R2

and8loop:
	LDREXB	(R1), R6

	DMB	MB_ISHST

	AND 	R2, R6
	STREXB	R6, (R1), R0
	CMP	$0, R0
	BNE	and8loop

	DMB	MB_ISH

	RET

TEXT armOr8<>(SB),NOSPLIT,$0-5
	// addr is already in R1
	MOVB	v+4(FP), R2

or8loop:
	LDREXB	(R1), R6

	DMB	MB_ISHST

	ORR 	R2, R6
	STREXB	R6, (R1), R0
	CMP	$0, R0
	BNE	or8loop

	DMB	MB_ISH

	RET

// The following functions all panic if their address argument isn't
// 8-byte aligned. Since we're calling back into Go code to do this,
// we have to cooperate with stack unwinding. In the normal case, the
// functions tail-call into the appropriate implementation, which
// means they must not open a frame. Hence, when they go down the
// panic path, at that point they push the LR to create a real frame
// (they don't need to pop it because panic won't return; however, we
// do need to set the SP delta back).

// Check if R1 is 8-byte aligned, panic if not.
// Clobbers R2.
#define CHECK_ALIGN \
	AND.S	$7, R1, R2 \
	BEQ 	4(PC) \
	MOVW.W	R14, -4(R13) /* prepare a real frame */ \
	BL	·panicUnaligned(SB) \
	ADD	$4, R13 /* compensate SP delta */

TEXT ·Cas64(SB),NOSPLIT,$-4-21
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1
	CHECK_ALIGN

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goCas64(SB)
#endif
	JMP	armCas64<>(SB)

TEXT ·Xadd64(SB),NOSPLIT,$-4-20
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1
	CHECK_ALIGN

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goXadd64(SB)
#endif
	JMP	armXadd64<>(SB)

TEXT ·Xchg64(SB),NOSPLIT,$-4-20
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1
	CHECK_ALIGN

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goXchg64(SB)
#endif
	JMP	armXchg64<>(SB)

TEXT ·Load64(SB),NOSPLIT,$-4-12
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1
	CHECK_ALIGN

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goLoad64(SB)
#endif
	JMP	armLoad64<>(SB)

TEXT ·Store64(SB),NOSPLIT,$-4-12
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1
	CHECK_ALIGN

#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goStore64(SB)
#endif
	JMP	armStore64<>(SB)

TEXT ·And8(SB),NOSPLIT,$-4-5
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1

// Uses STREXB/LDREXB that is armv6k or later.
// For simplicity we only enable this on armv7.
#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goAnd8(SB)
#endif
	JMP	armAnd8<>(SB)

TEXT ·Or8(SB),NOSPLIT,$-4-5
	NO_LOCAL_POINTERS
	MOVW	addr+0(FP), R1

// Uses STREXB/LDREXB that is armv6k or later.
// For simplicity we only enable this on armv7.
#ifndef GOARM_7
	MOVB	internal∕cpu·ARM+const_offsetARMHasV7Atomics(SB), R11
	CMP	$1, R11
	BEQ	2(PC)
	JMP	·goOr8(SB)
#endif
	JMP	armOr8<>(SB)
