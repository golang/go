// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !race

#include "textflag.h"

// ARM atomic operations, for use by asm_$(GOOS)_arm.s.

#define DMB_ISHST_7 \
	MOVB	runtime·goarm(SB), R11; \
	CMP	$7, R11; \
	BLT	2(PC); \
	WORD	$0xf57ff05a	// dmb ishst

#define DMB_ISH_7 \
	MOVB	runtime·goarm(SB), R11; \
	CMP	$7, R11; \
	BLT	2(PC); \
	WORD	$0xf57ff05b	// dmb ish

TEXT ·armCompareAndSwapUint32(SB),NOSPLIT,$0-13
	MOVW	addr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casloop:
	// LDREX and STREX were introduced in ARMv6.
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail
	DMB_ISHST_7
	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casloop
	MOVW	$1, R0
	DMB_ISH_7
	MOVBU	R0, swapped+12(FP)
	RET
casfail:
	MOVW	$0, R0
	MOVBU	R0, swapped+12(FP)
	RET

TEXT ·armCompareAndSwapUint64(SB),NOSPLIT,$0-21
	BL	fastCheck64<>(SB)
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)
	MOVW	old_lo+4(FP), R2
	MOVW	old_hi+8(FP), R3
	MOVW	new_lo+12(FP), R4
	MOVW	new_hi+16(FP), R5
cas64loop:
	// LDREXD and STREXD were introduced in ARMv6k.
	LDREXD	(R1), R6  // loads R6 and R7
	CMP	R2, R6
	BNE	cas64fail
	CMP	R3, R7
	BNE	cas64fail
	DMB_ISHST_7
	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	cas64loop
	MOVW	$1, R0
	DMB_ISH_7
	MOVBU	R0, swapped+20(FP)
	RET
cas64fail:
	MOVW	$0, R0
	MOVBU	R0, swapped+20(FP)
	RET

TEXT ·armAddUint32(SB),NOSPLIT,$0-12
	MOVW	addr+0(FP), R1
	MOVW	delta+4(FP), R2
addloop:
	// LDREX and STREX were introduced in ARMv6.
	LDREX	(R1), R3
	ADD	R2, R3
	DMB_ISHST_7
	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	addloop
	DMB_ISH_7
	MOVW	R3, new+8(FP)
	RET

TEXT ·armAddUint64(SB),NOSPLIT,$0-20
	BL	fastCheck64<>(SB)
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)
	MOVW	delta_lo+4(FP), R2
	MOVW	delta_hi+8(FP), R3
add64loop:
	// LDREXD and STREXD were introduced in ARMv6k.
	LDREXD	(R1), R4	// loads R4 and R5
	ADD.S	R2, R4
	ADC	R3, R5
	DMB_ISHST_7
	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	add64loop
	DMB_ISH_7
	MOVW	R4, new_lo+12(FP)
	MOVW	R5, new_hi+16(FP)
	RET

TEXT ·armSwapUint32(SB),NOSPLIT,$0-12
	MOVW	addr+0(FP), R1
	MOVW	new+4(FP), R2
swaploop:
	// LDREX and STREX were introduced in ARMv6.
	LDREX	(R1), R3
	DMB_ISHST_7
	STREX	R2, (R1), R0
	CMP	$0, R0
	BNE	swaploop
	DMB_ISH_7
	MOVW	R3, old+8(FP)
	RET

TEXT ·armSwapUint64(SB),NOSPLIT,$0-20
	BL	fastCheck64<>(SB)
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)
	MOVW	new_lo+4(FP), R2
	MOVW	new_hi+8(FP), R3
swap64loop:
	// LDREXD and STREXD were introduced in ARMv6k.
	LDREXD	(R1), R4	// loads R4 and R5
	DMB_ISHST_7
	STREXD	R2, (R1), R0	// stores R2 and R3
	CMP	$0, R0
	BNE	swap64loop
	DMB_ISH_7
	MOVW	R4, old_lo+12(FP)
	MOVW	R5, old_hi+16(FP)
	RET

TEXT ·armLoadUint64(SB),NOSPLIT,$0-12
	BL	fastCheck64<>(SB)
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)
load64loop:
	LDREXD	(R1), R2	// loads R2 and R3
	DMB_ISHST_7
	STREXD	R2, (R1), R0	// stores R2 and R3
	CMP	$0, R0
	BNE	load64loop
	DMB_ISH_7
	MOVW	R2, val_lo+4(FP)
	MOVW	R3, val_hi+8(FP)
	RET

TEXT ·armStoreUint64(SB),NOSPLIT,$0-12
	BL	fastCheck64<>(SB)
	MOVW	addr+0(FP), R1
	// make unaligned atomic access panic
	AND.S	$7, R1, R2
	BEQ 	2(PC)
	MOVW	R2, (R2)
	MOVW	val_lo+4(FP), R2
	MOVW	val_hi+8(FP), R3
store64loop:
	LDREXD	(R1), R4	// loads R4 and R5
	DMB_ISHST_7
	STREXD	R2, (R1), R0	// stores R2 and R3
	CMP	$0, R0
	BNE	store64loop
	DMB_ISH_7
	RET

// Check for broken 64-bit LDREXD as found in QEMU.
// LDREXD followed by immediate STREXD should succeed.
// If it fails, try a few times just to be sure (maybe our thread got
// rescheduled between the two instructions) and then panic.
// A bug in some copies of QEMU makes STREXD never succeed,
// which will make uses of the 64-bit atomic operations loop forever.
// If things are working, set okLDREXD to avoid future checks.
// https://bugs.launchpad.net/qemu/+bug/670883.
TEXT	check64<>(SB),NOSPLIT,$16-0
	MOVW	$10, R1
	// 8-aligned stack address scratch space.
	MOVW	$8(R13), R5
	AND	$~7, R5
loop:
	LDREXD	(R5), R2
	STREXD	R2, (R5), R0
	CMP	$0, R0
	BEQ	ok
	SUB	$1, R1
	CMP	$0, R1
	BNE	loop
	// Must be buggy QEMU.
	BL	·panic64(SB)
ok:
	RET

// Fast, cached version of check. No frame, just MOVW CMP RET after first time.
TEXT	fastCheck64<>(SB),NOSPLIT,$-4
	MOVW	ok64<>(SB), R0
	CMP	$0, R0	// have we been here before?
	RET.NE
	B	slowCheck64<>(SB)

TEXT slowCheck64<>(SB),NOSPLIT,$0-0
	BL	check64<>(SB)
	// Still here, must be okay.
	MOVW	$1, R0
	MOVW	R0, ok64<>(SB)
	RET

GLOBL ok64<>(SB), NOPTR, $4
