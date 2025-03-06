// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips64 || mips64le

#include "textflag.h"

#define SYNC	WORD $0xf

// func cas(ptr *uint32, old, new uint32) bool
// Atomically:
//	if *ptr == old {
//		*ptr = new
//		return true
//	} else {
//		return false
//	}
TEXT ·Cas(SB), NOSPLIT, $0-17
	MOVV	ptr+0(FP), R1
	MOVW	old+8(FP), R2
	MOVW	new+12(FP), R5
	SYNC
cas_again:
	MOVV	R5, R3
	LL	(R1), R4
	BNE	R2, R4, cas_fail
	SC	R3, (R1)
	BEQ	R3, cas_again
	MOVV	$1, R1
	MOVB	R1, ret+16(FP)
	SYNC
	RET
cas_fail:
	MOVV	$0, R1
	JMP	-4(PC)

// func	Cas64(ptr *uint64, old, new uint64) bool
// Atomically:
//	if *ptr == old {
//		*ptr = new
//		return true
//	} else {
//		return false
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOVV	ptr+0(FP), R1
	MOVV	old+8(FP), R2
	MOVV	new+16(FP), R5
	SYNC
cas64_again:
	MOVV	R5, R3
	LLV	(R1), R4
	BNE	R2, R4, cas64_fail
	SCV	R3, (R1)
	BEQ	R3, cas64_again
	MOVV	$1, R1
	MOVB	R1, ret+24(FP)
	SYNC
	RET
cas64_fail:
	MOVV	$0, R1
	JMP	-4(PC)

TEXT ·Casint32(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

TEXT ·Casint64(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·Casuintptr(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·CasRel(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

TEXT ·Loaduintptr(SB),  NOSPLIT|NOFRAME, $0-16
	JMP	·Load64(SB)

TEXT ·Loaduint(SB), NOSPLIT|NOFRAME, $0-16
	JMP	·Load64(SB)

TEXT ·Storeint32(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·Storeint64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·Storeuintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

TEXT ·Loadint32(SB), NOSPLIT, $0-12
	JMP	·Load(SB)

TEXT ·Loadint64(SB), NOSPLIT, $0-16
	JMP	·Load64(SB)

TEXT ·Xaddint32(SB), NOSPLIT, $0-20
	JMP	·Xadd(SB)

TEXT ·Xaddint64(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

// func Casp1(val *unsafe.Pointer, old, new unsafe.Pointer) bool
// Atomically:
//	if *val == old {
//		*val = new
//		return true
//	} else {
//		return false
//	}
TEXT ·Casp1(SB), NOSPLIT, $0-25
	JMP ·Cas64(SB)

// uint32 xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R2
	MOVW	delta+8(FP), R3
	SYNC
	LL	(R2), R1
	ADDU	R1, R3, R4
	MOVV	R4, R1
	SC	R4, (R2)
	BEQ	R4, -4(PC)
	MOVW	R1, ret+16(FP)
	SYNC
	RET

// uint64 Xadd64(uint64 volatile *ptr, int64 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R2
	MOVV	delta+8(FP), R3
	SYNC
	LLV	(R2), R1
	ADDVU	R1, R3, R4
	MOVV	R4, R1
	SCV	R4, (R2)
	BEQ	R4, -4(PC)
	MOVV	R1, ret+16(FP)
	SYNC
	RET

// uint8 Xchg(ptr *uint8, new uint8)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg8(SB), NOSPLIT, $0-17
	MOVV	ptr+0(FP), R2
	MOVBU	new+8(FP), R5
#ifdef GOARCH_mips64
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R2
#endif
	// R4 = ((ptr & 3) * 8)
	AND	$3, R2, R4
	SLLV	$3, R4
	// Shift val for aligned ptr. R7 = (0xFF << R4) ^ (-1)
	MOVV	$0xFF, R7
	SLLV	R4, R7
	XOR	$-1, R7
	AND	$~3, R2
	SLLV	R4, R5

	SYNC
	LL	(R2), R9
	AND	R7, R9, R8
	OR	R5, R8
	SC	R8, (R2)
	BEQ	R8, -5(PC)
	SYNC
	SRLV	R4, R9
	MOVBU	R9, ret+16(FP)
	RET

// uint32 Xchg(ptr *uint32, new uint32)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R2
	MOVW	new+8(FP), R5

	SYNC
	MOVV	R5, R3
	LL	(R2), R1
	SC	R3, (R2)
	BEQ	R3, -3(PC)
	MOVW	R1, ret+16(FP)
	SYNC
	RET

// uint64 Xchg64(ptr *uint64, new uint64)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R2
	MOVV	new+8(FP), R5

	SYNC
	MOVV	R5, R3
	LLV	(R2), R1
	SCV	R3, (R2)
	BEQ	R3, -3(PC)
	MOVV	R1, ret+16(FP)
	SYNC
	RET

TEXT ·Xchgint32(SB), NOSPLIT, $0-20
	JMP	·Xchg(SB)

TEXT ·Xchgint64(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

TEXT ·StorepNoWB(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreRel(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·StoreRel64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·Store(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R1
	MOVW	val+8(FP), R2
	SYNC
	MOVW	R2, 0(R1)
	SYNC
	RET

TEXT ·Store8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R1
	MOVB	val+8(FP), R2
	SYNC
	MOVB	R2, 0(R1)
	SYNC
	RET

TEXT ·Store64(SB), NOSPLIT, $0-16
	MOVV	ptr+0(FP), R1
	MOVV	val+8(FP), R2
	SYNC
	MOVV	R2, 0(R1)
	SYNC
	RET

// void	Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R1
	MOVBU	val+8(FP), R2
	// Align ptr down to 4 bytes so we can use 32-bit load/store.
	MOVV	$~3, R3
	AND	R1, R3
	// Compute val shift.
#ifdef GOARCH_mips64
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R1
#endif
	// R4 = ((ptr & 3) * 8)
	AND	$3, R1, R4
	SLLV	$3, R4
	// Shift val for aligned ptr. R2 = val << R4
	SLLV	R4, R2

	SYNC
	LL	(R3), R4
	OR	R2, R4
	SC	R4, (R3)
	BEQ	R4, -4(PC)
	SYNC
	RET

// void	And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R1
	MOVBU	val+8(FP), R2
	// Align ptr down to 4 bytes so we can use 32-bit load/store.
	MOVV	$~3, R3
	AND	R1, R3
	// Compute val shift.
#ifdef GOARCH_mips64
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R1
#endif
	// R4 = ((ptr & 3) * 8)
	AND	$3, R1, R4
	SLLV	$3, R4
	// Shift val for aligned ptr. R2 = val << R4 | ^(0xFF << R4)
	MOVV	$0xFF, R5
	SLLV	R4, R2
	SLLV	R4, R5
	NOR	R0, R5
	OR	R5, R2

	SYNC
	LL	(R3), R4
	AND	R2, R4
	SC	R4, (R3)
	BEQ	R4, -4(PC)
	SYNC
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R1
	MOVW	val+8(FP), R2

	SYNC
	LL	(R1), R3
	OR	R2, R3
	SC	R3, (R1)
	BEQ	R3, -4(PC)
	SYNC
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R1
	MOVW	val+8(FP), R2

	SYNC
	LL	(R1), R3
	AND	R2, R3
	SC	R3, (R1)
	BEQ	R3, -4(PC)
	SYNC
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R1
	MOVW	val+8(FP), R2

	SYNC
	LL	(R1), R3
	OR	R2, R3, R4
	SC	R4, (R1)
	BEQ	R4, -3(PC)
	SYNC
	MOVW	R3, ret+16(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R1
	MOVW	val+8(FP), R2

	SYNC
	LL	(R1), R3
	AND	R2, R3, R4
	SC	R4, (R1)
	BEQ	R4, -3(PC)
	SYNC
	MOVW	R3, ret+16(FP)
	RET

// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R1
	MOVV	val+8(FP), R2

	SYNC
	LLV	(R1), R3
	OR	R2, R3, R4
	SCV	R4, (R1)
	BEQ	R4, -3(PC)
	SYNC
	MOVV	R3, ret+16(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R1
	MOVV	val+8(FP), R2

	SYNC
	LLV	(R1), R3
	AND	R2, R3, R4
	SCV	R4, (R1)
	BEQ	R4, -3(PC)
	SYNC
	MOVV	R3, ret+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-24
	JMP	·And64(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-24
	JMP	·Or64(SB)

// uint32 ·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$0-12
	MOVV	ptr+0(FP), R1
	SYNC
	MOVWU	0(R1), R1
	SYNC
	MOVW	R1, ret+8(FP)
	RET

// uint8 ·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$0-9
	MOVV	ptr+0(FP), R1
	SYNC
	MOVBU	0(R1), R1
	SYNC
	MOVB	R1, ret+8(FP)
	RET

// uint64 ·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R1
	SYNC
	MOVV	0(R1), R1
	SYNC
	MOVV	R1, ret+8(FP)
	RET

// void *·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R1
	SYNC
	MOVV	0(R1), R1
	SYNC
	MOVV	R1, ret+8(FP)
	RET

// uint32 ·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-12
	JMP	atomic·Load(SB)

// uint64 ·LoadAcq64(uint64 volatile* ptr)
TEXT ·LoadAcq64(SB),NOSPLIT|NOFRAME,$0-16
	JMP	atomic·Load64(SB)

// uintptr ·LoadAcquintptr(uintptr volatile* ptr)
TEXT ·LoadAcquintptr(SB),NOSPLIT|NOFRAME,$0-16
	JMP	atomic·Load64(SB)
