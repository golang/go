// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips || mipsle

#include "textflag.h"

// func Cas(val *int32, old, new int32) bool
// Atomically:
//	if *val == old {
//		*val = new
//		return true
//	} else {
//		return false
//	}
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
	SYNC
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

// uint32 Xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
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

// uint32 Xchg(ptr *uint32, new uint32)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
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

// uint8 Xchg(ptr *uint8, new uint8)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg8(SB), NOSPLIT, $0-9
	MOVW	ptr+0(FP), R2
	MOVBU	new+4(FP), R5
#ifdef GOARCH_mips
	// Big endian.  ptr = ptr ^ 3
	XOR	$3, R2
#endif
	// R4 = ((ptr & 3) * 8)
	AND	$3, R2, R4
	SLL	$3, R4
	// Shift val for aligned ptr. R7 = (0xFF << R4) ^ (-1)
	MOVW	$0xFF, R7
	SLL	R4, R7
	XOR	$-1, R7
	AND	$~3, R2
	SLL	R4, R5

	SYNC
	LL	(R2), R9
	AND	R7, R9, R8
	OR	R5, R8
	SC	R8, (R2)
	BEQ	R8, -5(PC)
	SYNC
	SRL	R4, R9
	MOVBU	R9, ret+8(FP)
	RET

TEXT ·Casint32(SB),NOSPLIT,$0-13
	JMP	·Cas(SB)

TEXT ·Casint64(SB),NOSPLIT,$0-21
	JMP	·Cas64(SB)

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

TEXT ·Storeint32(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

TEXT ·Storeint64(SB),NOSPLIT,$0-12
	JMP	·Store64(SB)

TEXT ·Storeuintptr(SB),NOSPLIT,$0-8
	JMP	·Store(SB)

TEXT ·Xadduintptr(SB),NOSPLIT,$0-12
	JMP	·Xadd(SB)

TEXT ·Loadint32(SB),NOSPLIT,$0-8
	JMP	·Load(SB)

TEXT ·Loadint64(SB),NOSPLIT,$0-12
	JMP	·Load64(SB)

TEXT ·Xaddint32(SB),NOSPLIT,$0-12
	JMP	·Xadd(SB)

TEXT ·Xaddint64(SB),NOSPLIT,$0-20
	JMP	·Xadd64(SB)

TEXT ·Casp1(SB),NOSPLIT,$0-13
	JMP	·Cas(SB)

TEXT ·Xchgint32(SB),NOSPLIT,$0-12
	JMP	·Xchg(SB)

TEXT ·Xchgint64(SB),NOSPLIT,$0-20
	JMP	·Xchg64(SB)

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

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-12
	MOVW	ptr+0(FP), R1
	MOVW	val+4(FP), R2

	SYNC
	LL	(R1), R3
	OR	R2, R3, R4
	SC	R4, (R1)
	BEQ	R4, -4(PC)
	SYNC
	MOVW	R3, ret+8(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-12
	MOVW	ptr+0(FP), R1
	MOVW	val+4(FP), R2

	SYNC
	LL	(R1), R3
	AND	R2, R3, R4
	SC	R4, (R1)
	BEQ	R4, -4(PC)
	SYNC
	MOVW	R3, ret+8(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-12
	JMP	·And32(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-12
	JMP	·Or32(SB)

TEXT ·spinLock(SB),NOSPLIT,$0-4
	MOVW	state+0(FP), R1
	MOVW	$1, R2
	SYNC
try_lock:
	MOVW	R2, R3
check_again:
	LL	(R1), R4
	BNE	R4, check_again
	SC	R3, (R1)
	BEQ	R3, try_lock
	SYNC
	RET

TEXT ·spinUnlock(SB),NOSPLIT,$0-4
	MOVW	state+0(FP), R1
	SYNC
	MOVW	R0, (R1)
	SYNC
	RET
