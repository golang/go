// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// bool cas(uint32 *ptr, uint32 old, uint32 new)
// Atomically:
//	if(*ptr == old){
//		*ptr = new;
//		return 1;
//	} else
//		return 0;
TEXT ·Cas(SB), NOSPLIT, $0-17
	MOVV	ptr+0(FP), R4
	MOVW	old+8(FP), R5
	MOVW	new+12(FP), R6
	DBAR
cas_again:
	MOVV	R6, R7
	LL	(R4), R8
	BNE	R5, R8, cas_fail
	SC	R7, (R4)
	BEQ	R7, cas_again
	MOVV	$1, R4
	MOVB	R4, ret+16(FP)
	DBAR
	RET
cas_fail:
	MOVV	$0, R4
	JMP	-4(PC)

// bool	cas64(uint64 *ptr, uint64 old, uint64 new)
// Atomically:
//	if(*ptr == old){
//		*ptr = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOVV	ptr+0(FP), R4
	MOVV	old+8(FP), R5
	MOVV	new+16(FP), R6
	DBAR
cas64_again:
	MOVV	R6, R7
	LLV	(R4), R8
	BNE	R5, R8, cas64_fail
	SCV	R7, (R4)
	BEQ	R7, cas64_again
	MOVV	$1, R4
	MOVB	R4, ret+24(FP)
	DBAR
	RET
cas64_fail:
	MOVV	$0, R4
	JMP	-4(PC)

TEXT ·Casuintptr(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·CasRel(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

TEXT ·Loaduintptr(SB),  NOSPLIT|NOFRAME, $0-16
	JMP	·Load64(SB)

TEXT ·Loaduint(SB), NOSPLIT|NOFRAME, $0-16
	JMP	·Load64(SB)

TEXT ·Storeuintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

TEXT ·Loadint64(SB), NOSPLIT, $0-16
	JMP	·Load64(SB)

TEXT ·Xaddint64(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

// bool casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT ·Casp1(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

// uint32 xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	delta+8(FP), R5
	DBAR
	LL	(R4), R6
	ADDU	R6, R5, R7
	MOVV	R7, R6
	SC	R7, (R4)
	BEQ	R7, -4(PC)
	MOVW	R6, ret+16(FP)
	DBAR
	RET

TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	delta+8(FP), R5
	DBAR
	LLV	(R4), R6
	ADDVU	R6, R5, R7
	MOVV	R7, R6
	SCV	R7, (R4)
	BEQ	R7, -4(PC)
	MOVV	R6, ret+16(FP)
	DBAR
	RET

TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	new+8(FP), R5

	DBAR
	MOVV	R5, R6
	LL	(R4), R7
	SC	R6, (R4)
	BEQ	R6, -3(PC)
	MOVW	R7, ret+16(FP)
	DBAR
	RET

TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	new+8(FP), R5

	DBAR
	MOVV	R5, R6
	LLV	(R4), R7
	SCV	R6, (R4)
	BEQ	R6, -3(PC)
	MOVV	R7, ret+16(FP)
	DBAR
	RET

TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

TEXT ·StorepNoWB(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreRel(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·StoreRel64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-16
	JMP     ·Store64(SB)

TEXT ·Store(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	DBAR
	MOVW	R5, 0(R4)
	DBAR
	RET

TEXT ·Store8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVB	val+8(FP), R5
	DBAR
	MOVB	R5, 0(R4)
	DBAR
	RET

TEXT ·Store64(SB), NOSPLIT, $0-16
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	DBAR
	MOVV	R5, 0(R4)
	DBAR
	RET

// void	Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVBU	val+8(FP), R5
	// Align ptr down to 4 bytes so we can use 32-bit load/store.
	MOVV	$~3, R6
	AND	R4, R6
	// R7 = ((ptr & 3) * 8)
	AND	$3, R4, R7
	SLLV	$3, R7
	// Shift val for aligned ptr. R5 = val << R4
	SLLV	R7, R5

	DBAR
	LL	(R6), R7
	OR	R5, R7
	SC	R7, (R6)
	BEQ	R7, -4(PC)
	DBAR
	RET

// void	And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVBU	val+8(FP), R5
	// Align ptr down to 4 bytes so we can use 32-bit load/store.
	MOVV	$~3, R6
	AND	R4, R6
	// R7 = ((ptr & 3) * 8)
	AND	$3, R4, R7
	SLLV	$3, R7
	// Shift val for aligned ptr. R5 = val << R7 | ^(0xFF << R7)
	MOVV	$0xFF, R8
	SLLV	R7, R5
	SLLV	R7, R8
	NOR	R0, R8
	OR	R8, R5

	DBAR
	LL	(R6), R7
	AND	R5, R7
	SC	R7, (R6)
	BEQ	R7, -4(PC)
	DBAR
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	DBAR
	LL	(R4), R6
	OR	R5, R6
	SC	R6, (R4)
	BEQ	R6, -4(PC)
	DBAR
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	DBAR
	LL	(R4), R6
	AND	R5, R6
	SC	R6, (R4)
	BEQ	R6, -4(PC)
	DBAR
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	DBAR
	LL	(R4), R6
	OR	R5, R6, R7
	SC	R7, (R4)
	BEQ	R7, -4(PC)
	DBAR
	MOVW R6, ret+16(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	DBAR
	LL	(R4), R6
	AND	R5, R6, R7
	SC	R7, (R4)
	BEQ	R7, -4(PC)
	DBAR
	MOVW R6, ret+16(FP)
	RET

// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	DBAR
	LLV	(R4), R6
	OR	R5, R6, R7
	SCV	R7, (R4)
	BEQ	R7, -4(PC)
	DBAR
	MOVV R6, ret+16(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	DBAR
	LLV	(R4), R6
	AND	R5, R6, R7
	SCV	R7, (R4)
	BEQ	R7, -4(PC)
	DBAR
	MOVV R6, ret+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-24
	JMP	·And64(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-24
	JMP	·Or64(SB)

// uint32 runtime∕internal∕atomic·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$0-12
	MOVV	ptr+0(FP), R19
	DBAR
	MOVWU	0(R19), R19
	DBAR
	MOVW	R19, ret+8(FP)
	RET

// uint8 runtime∕internal∕atomic·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$0-9
	MOVV	ptr+0(FP), R19
	DBAR
	MOVBU	0(R19), R19
	DBAR
	MOVB	R19, ret+8(FP)
	RET

// uint64 runtime∕internal∕atomic·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R19
	DBAR
	MOVV	0(R19), R19
	DBAR
	MOVV	R19, ret+8(FP)
	RET

// void *runtime∕internal∕atomic·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R19
	DBAR
	MOVV	0(R19), R19
	DBAR
	MOVV	R19, ret+8(FP)
	RET

// uint32 runtime∕internal∕atomic·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-12
	JMP	·Load(SB)

// uint64 ·LoadAcq64(uint64 volatile* ptr)
TEXT ·LoadAcq64(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

// uintptr ·LoadAcquintptr(uintptr volatile* ptr)
TEXT ·LoadAcquintptr(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

