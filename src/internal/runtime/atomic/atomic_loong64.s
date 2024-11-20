// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
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

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLAMCAS(SB), R8
	BEQ	R8, cas_again
	MOVV	R5, R7  // backup old value
	AMCASDBW	R6, (R4), R5
	BNE	R7, R5, cas_fail0
	MOVV	$1, R4
	MOVB	R4, ret+16(FP)
	RET
cas_fail0:
	MOVB	R0, ret+16(FP)
	RET

	// Implemented using the ll-sc instruction pair
	DBAR	$0x14	// LoadAcquire barrier
cas_again:
	MOVV	R6, R7
	LL	(R4), R8
	BNE	R5, R8, cas_fail1
	SC	R7, (R4)
	BEQ	R7, cas_again
	MOVV	$1, R4
	MOVB	R4, ret+16(FP)
	DBAR	$0x12	// StoreRelease barrier
	RET
cas_fail1:
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

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLAMCAS(SB), R8
	BEQ	R8, cas64_again
	MOVV	R5, R7  // backup old value
	AMCASDBV	R6, (R4), R5
	BNE	R7, R5, cas64_fail0
	MOVV	$1, R4
	MOVB	R4, ret+24(FP)
	RET
cas64_fail0:
	MOVB	R0, ret+24(FP)
	RET

	// Implemented using the ll-sc instruction pair
	DBAR	$0x14
cas64_again:
	MOVV	R6, R7
	LLV	(R4), R8
	BNE	R5, R8, cas64_fail1
	SCV	R7, (R4)
	BEQ	R7, cas64_again
	MOVV	$1, R4
	MOVB	R4, ret+24(FP)
	DBAR	$0x12
	RET
cas64_fail1:
	MOVV	$0, R4
	JMP	-4(PC)

TEXT ·Casint32(SB),NOSPLIT,$0-17
	JMP	·Cas(SB)

TEXT ·Casint64(SB),NOSPLIT,$0-25
	JMP	·Cas64(SB)

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

TEXT ·Xaddint32(SB),NOSPLIT,$0-20
	JMP	·Xadd(SB)

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

// uint32 Xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	delta+8(FP), R5
	AMADDDBW	R5, (R4), R6
	ADDV	R6, R5, R4
	MOVW	R4, ret+16(FP)
	RET

// func Xadd64(ptr *uint64, delta int64) uint64
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	delta+8(FP), R5
	AMADDDBV	R5, (R4), R6
	ADDV	R6, R5, R4
	MOVV	R4, ret+16(FP)
	RET

// uint8 Xchg8(ptr *uint8, new uint8)
// Atomically:
//     old := *ptr;
//     *ptr = new;
//     return old;
TEXT ·Xchg8(SB), NOSPLIT, $0-17
	MOVV	ptr+0(FP), R4
	MOVBU	new+8(FP), R5

	// R6 = ((ptr & 3) * 8)
	AND	$3, R4, R6
	SLLV	$3, R6

	// R7 = ((0xFF) << R6) ^ (-1)
	MOVV	$0xFF, R8
	SLLV	R6, R8, R7
	XOR	$-1, R7

	// R4 = ptr & (~3)
	MOVV	$~3, R8
	AND	R8, R4

	// R5 = ((val) << R6)
	SLLV	R6, R5

	DBAR	$0x14	// LoadAcquire barrier
_xchg8_again:
	LL	(R4), R8
	MOVV	R8, R9	// backup old val
	AND	R7, R8
	OR	R5, R8
	SC	R8, (R4)
	BEQ	R8, _xchg8_again
	DBAR	$0x12	// StoreRelease barrier
	SRLV	R6, R9, R9
	MOVBU	R9, ret+16(FP)
	RET

// func Xchg(ptr *uint32, new uint32) uint32
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	new+8(FP), R5
	AMSWAPDBW	R5, (R4), R6
	MOVW	R6, ret+16(FP)
	RET

// func Xchg64(ptr *uint64, new uint64) uint64
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	new+8(FP), R5
	AMSWAPDBV	R5, (R4), R6
	MOVV	R6, ret+16(FP)
	RET

TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

// func Xchgint32(ptr *int32, new int32) int32
TEXT ·Xchgint32(SB), NOSPLIT, $0-20
	JMP	·Xchg(SB)

// func Xchgint64(ptr *int64, new int64) int64
TEXT ·Xchgint64(SB), NOSPLIT, $0-24
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
	AMSWAPDBW	R5, (R4), R0
	RET

TEXT ·Store8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVB	val+8(FP), R5
	MOVBU	internal∕cpu·Loong64+const_offsetLoong64HasLAM_BH(SB), R6
	BEQ	R6, _legacy_store8_
	AMSWAPDBB	R5, (R4), R0
	RET
_legacy_store8_:
	// StoreRelease barrier
	DBAR	$0x12
	MOVB	R5, 0(R4)
	DBAR	$0x18
	RET

TEXT ·Store64(SB), NOSPLIT, $0-16
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	AMSWAPDBV	R5, (R4), R0
	RET

// void	Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVBU	val+8(FP), R5
	// R6 = ptr & (~3)
	MOVV	$~3, R6
	AND	R4, R6
	// R7 = ((ptr & 3) * 8)
	AND	$3, R4, R7
	SLLV	$3, R7
	// R5 = val << R7
	SLLV	R7, R5
	AMORDBW	R5, (R6), R0
	RET

// void	And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVV	ptr+0(FP), R4
	MOVBU	val+8(FP), R5
	// R6 = ptr & (~3)
	MOVV	$~3, R6
	AND	R4, R6
	// R7 = ((ptr & 3) * 8)
	AND	$3, R4, R7
	SLLV	$3, R7
	// R5 = ((val ^ 0xFF) << R7) ^ (-1)
	XOR	$255, R5
	SLLV	R7,  R5
	XOR	$-1, R5
	AMANDDBW	R5, (R6), R0
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	AMORDBW	R5, (R4), R0
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	AMANDDBW	R5, (R4), R0
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	AMORDBW R5, (R4), R6
	MOVW	R6, ret+16(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-20
	MOVV	ptr+0(FP), R4
	MOVW	val+8(FP), R5
	AMANDDBW	R5, (R4), R6
	MOVW	R6, ret+16(FP)
	RET

// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	AMORDBV	R5, (R4), R6
	MOVV	R6, ret+16(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-24
	MOVV	ptr+0(FP), R4
	MOVV	val+8(FP), R5
	AMANDDBV	R5, (R4), R6
	MOVV	R6, ret+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-24
	JMP	·And64(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-24
	JMP	·Or64(SB)

// uint32 internal∕runtime∕atomic·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$0-12
	MOVV	ptr+0(FP), R19
	MOVWU	0(R19), R19
	DBAR	$0x14	// LoadAcquire barrier
	MOVW	R19, ret+8(FP)
	RET

// uint8 internal∕runtime∕atomic·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$0-9
	MOVV	ptr+0(FP), R19
	MOVBU	0(R19), R19
	DBAR	$0x14
	MOVB	R19, ret+8(FP)
	RET

// uint64 internal∕runtime∕atomic·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R19
	MOVV	0(R19), R19
	DBAR	$0x14
	MOVV	R19, ret+8(FP)
	RET

// void *internal∕runtime∕atomic·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$0-16
	JMP     ·Load64(SB)

// uint32 internal∕runtime∕atomic·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-12
	JMP	·Load(SB)

// uint64 ·LoadAcq64(uint64 volatile* ptr)
TEXT ·LoadAcq64(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

// uintptr ·LoadAcquintptr(uintptr volatile* ptr)
TEXT ·LoadAcquintptr(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

