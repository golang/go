// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ARM atomic operations, for use by asm_$(GOOS)_arm.s.

TEXT ·armCompareAndSwapUint32(SB),7,$0
	MOVW	valptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casloop:
	// LDREX and STREX were introduced in ARM 6.
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail
	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casloop
	MOVW	$1, R0
	MOVBU	R0, ret+12(FP)
	RET
casfail:
	MOVW	$0, R0
	MOVBU	R0, ret+12(FP)
	RET

TEXT ·armCompareAndSwapUint64(SB),7,$0
	BL	fastCheck64<>(SB)
	MOVW	valptr+0(FP), R1
	MOVW	oldlo+4(FP), R2
	MOVW	oldhi+8(FP), R3
	MOVW	newlo+12(FP), R4
	MOVW	newhi+16(FP), R5
cas64loop:
	// LDREXD and STREXD were introduced in ARM 11.
	LDREXD	(R1), R6  // loads R6 and R7
	CMP	R2, R6
	BNE	cas64fail
	CMP	R3, R7
	BNE	cas64fail
	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	cas64loop
	MOVW	$1, R0
	MOVBU	R0, ret+20(FP)
	RET
cas64fail:
	MOVW	$0, R0
	MOVBU	R0, ret+20(FP)
	RET

TEXT ·armAddUint32(SB),7,$0
	MOVW	valptr+0(FP), R1
	MOVW	delta+4(FP), R2
addloop:
	// LDREX and STREX were introduced in ARM 6.
	LDREX	(R1), R3
	ADD	R2, R3
	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	addloop
	MOVW	R3, ret+8(FP)
	RET

TEXT ·armAddUint64(SB),7,$0
	BL	fastCheck64<>(SB)
	MOVW	valptr+0(FP), R1
	MOVW	deltalo+4(FP), R2
	MOVW	deltahi+8(FP), R3
add64loop:
	// LDREXD and STREXD were introduced in ARM 11.
	LDREXD	(R1), R4	// loads R4 and R5
	ADD.S	R2, R4
	ADC	R3, R5
	STREXD	R4, (R1), R0	// stores R4 and R5
	CMP	$0, R0
	BNE	add64loop
	MOVW	R4, retlo+12(FP)
	MOVW	R5, rethi+16(FP)
	RET

// Check for broken 64-bit LDREXD as found in QEMU.
// LDREXD followed by immediate STREXD should succeed.
// If it fails, try a few times just to be sure (maybe our thread got
// rescheduled between the two instructions) and then panic.
// A bug in some copies of QEMU makes STREXD never succeed,
// which will make uses of the 64-bit atomic operations loop forever.
// If things are working, set okLDREXD to avoid future checks.
// https://bugs.launchpad.net/qemu/+bug/670883.
TEXT	check64<>(SB),7,$8
	MOVW	$10, R1
loop:
	LDREXD	(SP), R2
	STREXD	R2, (SP), R0
	CMP	$0, R0
	BEQ	ok
	SUB	$1, R1
	CMP	$0, R1
	BNE	loop
	// Must be buggy QEMU.
	BL	·panic64(SB)
ok:
	RET

// Fast, cached version of check.  No frame, just MOVW CMP RET after first time.
TEXT	fastCheck64<>(SB),7,$-4
	MOVW	ok64<>(SB), R0
	CMP	$0, R0	// have we been here before?
	RET.NE
	B	slowCheck64<>(SB)

TEXT slowCheck64<>(SB),7,$0
	BL	check64<>(SB)
	// Still here, must be okay.
	MOVW	$1, R0
	MOVW	R0, ok64<>(SB)
	RET

GLOBL ok64<>(SB), $4
