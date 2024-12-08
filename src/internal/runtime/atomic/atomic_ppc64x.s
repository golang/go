// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "textflag.h"

// For more details about how various memory models are
// enforced on POWER, the following paper provides more
// details about how they enforce C/C++ like models. This
// gives context about why the strange looking code
// sequences below work.
//
// http://www.rdrop.com/users/paulmck/scalability/paper/N2745r.2011.03.04a.html

// uint32 ·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$-8-12
	MOVD	ptr+0(FP), R3
	SYNC
	MOVWZ	0(R3), R3
	CMPW	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVW	R3, ret+8(FP)
	RET

// uint8 ·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$-8-9
	MOVD	ptr+0(FP), R3
	SYNC
	MOVBZ	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVB	R3, ret+8(FP)
	RET

// uint64 ·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$-8-16
	MOVD	ptr+0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RET

// void *·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$-8-16
	MOVD	ptr+0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RET

// uint32 ·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$-8-12
	MOVD   ptr+0(FP), R3
	MOVWZ  0(R3), R3
	CMPW   R3, R3, CR7
	BC     4, 30, 1(PC) // bne- cr7, 0x4
	ISYNC
	MOVW   R3, ret+8(FP)
	RET

// uint64 ·LoadAcq64(uint64 volatile* ptr)
TEXT ·LoadAcq64(SB),NOSPLIT|NOFRAME,$-8-16
	MOVD   ptr+0(FP), R3
	MOVD   0(R3), R3
	CMP    R3, R3, CR7
	BC     4, 30, 1(PC) // bne- cr7, 0x4
	ISYNC
	MOVD   R3, ret+8(FP)
	RET

// bool cas(uint32 *ptr, uint32 old, uint32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT ·Cas(SB), NOSPLIT, $0-17
	MOVD	ptr+0(FP), R3
	MOVWZ	old+8(FP), R4
	MOVWZ	new+12(FP), R5
	LWSYNC
cas_again:
	LWAR	(R3), R6
	CMPW	R6, R4
	BNE	cas_fail
	STWCCC	R5, (R3)
	BNE	cas_again
	MOVD	$1, R3
	LWSYNC
	MOVB	R3, ret+16(FP)
	RET
cas_fail:
	LWSYNC
	MOVB	R0, ret+16(FP)
	RET

// bool	·Cas64(uint64 *ptr, uint64 old, uint64 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOVD	ptr+0(FP), R3
	MOVD	old+8(FP), R4
	MOVD	new+16(FP), R5
	LWSYNC
cas64_again:
	LDAR	(R3), R6
	CMP	R6, R4
	BNE	cas64_fail
	STDCCC	R5, (R3)
	BNE	cas64_again
	MOVD	$1, R3
	LWSYNC
	MOVB	R3, ret+24(FP)
	RET
cas64_fail:
	LWSYNC
	MOVB	R0, ret+24(FP)
	RET

TEXT ·CasRel(SB), NOSPLIT, $0-17
	MOVD    ptr+0(FP), R3
	MOVWZ   old+8(FP), R4
	MOVWZ   new+12(FP), R5
	LWSYNC
cas_again:
	LWAR    (R3), $0, R6        // 0 = Mutex release hint
	CMPW    R6, R4
	BNE     cas_fail
	STWCCC  R5, (R3)
	BNE     cas_again
	MOVD    $1, R3
	MOVB    R3, ret+16(FP)
	RET
cas_fail:
	MOVB    R0, ret+16(FP)
	RET

TEXT ·Casint32(SB), NOSPLIT, $0-17
	BR	·Cas(SB)

TEXT ·Casint64(SB), NOSPLIT, $0-25
	BR	·Cas64(SB)

TEXT ·Casuintptr(SB), NOSPLIT, $0-25
	BR	·Cas64(SB)

TEXT ·Loaduintptr(SB),  NOSPLIT|NOFRAME, $0-16
	BR	·Load64(SB)

TEXT ·LoadAcquintptr(SB),  NOSPLIT|NOFRAME, $0-16
	BR	·LoadAcq64(SB)

TEXT ·Loaduint(SB), NOSPLIT|NOFRAME, $0-16
	BR	·Load64(SB)

TEXT ·Storeint32(SB), NOSPLIT, $0-12
	BR	·Store(SB)

TEXT ·Storeint64(SB), NOSPLIT, $0-16
	BR	·Store64(SB)

TEXT ·Storeuintptr(SB), NOSPLIT, $0-16
	BR	·Store64(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-16
	BR	·StoreRel64(SB)

TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	BR	·Xadd64(SB)

TEXT ·Loadint32(SB), NOSPLIT, $0-12
	BR	·Load(SB)

TEXT ·Loadint64(SB), NOSPLIT, $0-16
	BR	·Load64(SB)

TEXT ·Xaddint32(SB), NOSPLIT, $0-20
	BR	·Xadd(SB)

TEXT ·Xaddint64(SB), NOSPLIT, $0-24
	BR	·Xadd64(SB)

// bool casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT ·Casp1(SB), NOSPLIT, $0-25
	BR ·Cas64(SB)

// uint32 xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	delta+8(FP), R5
	LWSYNC
	LWAR	(R4), R3
	ADD	R5, R3
	STWCCC	R3, (R4)
	BNE	-3(PC)
	MOVW	R3, ret+16(FP)
	RET

// uint64 Xadd64(uint64 volatile *val, int64 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	delta+8(FP), R5
	LWSYNC
	LDAR	(R4), R3
	ADD	R5, R3
	STDCCC	R3, (R4)
	BNE	-3(PC)
	MOVD	R3, ret+16(FP)
	RET

// uint8 Xchg(ptr *uint8, new uint8)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg8(SB), NOSPLIT, $0-17
	MOVD	ptr+0(FP), R4
	MOVB	new+8(FP), R5
	LWSYNC
	LBAR	(R4), R3
	STBCCC	R5, (R4)
	BNE	-2(PC)
	ISYNC
	MOVB	R3, ret+16(FP)
	RET

// uint32 Xchg(ptr *uint32, new uint32)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	new+8(FP), R5
	LWSYNC
	LWAR	(R4), R3
	STWCCC	R5, (R4)
	BNE	-2(PC)
	ISYNC
	MOVW	R3, ret+16(FP)
	RET

// uint64 Xchg64(ptr *uint64, new uint64)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	new+8(FP), R5
	LWSYNC
	LDAR	(R4), R3
	STDCCC	R5, (R4)
	BNE	-2(PC)
	ISYNC
	MOVD	R3, ret+16(FP)
	RET

TEXT ·Xchgint32(SB), NOSPLIT, $0-20
	BR	·Xchg(SB)

TEXT ·Xchgint64(SB), NOSPLIT, $0-24
	BR	·Xchg64(SB)

TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	BR	·Xchg64(SB)

TEXT ·StorepNoWB(SB), NOSPLIT, $0-16
	BR	·Store64(SB)

TEXT ·Store(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	SYNC
	MOVW	R4, 0(R3)
	RET

TEXT ·Store8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVB	val+8(FP), R4
	SYNC
	MOVB	R4, 0(R3)
	RET

TEXT ·Store64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R3
	MOVD	val+8(FP), R4
	SYNC
	MOVD	R4, 0(R3)
	RET

TEXT ·StoreRel(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LWSYNC
	MOVW	R4, 0(R3)
	RET

TEXT ·StoreRel64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R3
	MOVD	val+8(FP), R4
	LWSYNC
	MOVD	R4, 0(R3)
	RET

// void ·Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	LWSYNC
again:
	LBAR	(R3), R6
	OR	R4, R6
	STBCCC	R6, (R3)
	BNE	again
	RET

// void ·And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	LWSYNC
again:
	LBAR	(R3), R6
	AND	R4, R6
	STBCCC	R6, (R3)
	BNE	again
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LWSYNC
again:
	LWAR	(R3), R6
	OR	R4, R6
	STWCCC	R6, (R3)
	BNE	again
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LWSYNC
again:
	LWAR	(R3),R6
	AND	R4, R6
	STWCCC	R6, (R3)
	BNE	again
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LWSYNC
again:
	LWAR	(R3), R6
	OR	R4, R6, R7
	STWCCC	R7, (R3)
	BNE	again
	MOVW	R6, ret+16(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LWSYNC
again:
	LWAR	(R3),R6
	AND	R4, R6, R7
	STWCCC	R7, (R3)
	BNE	again
	MOVW	R6, ret+16(FP)
	RET

// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R3
	MOVD	val+8(FP), R4
	LWSYNC
again:
	LDAR	(R3), R6
	OR	R4, R6, R7
	STDCCC	R7, (R3)
	BNE	again
	MOVD	R6, ret+16(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R3
	MOVD	val+8(FP), R4
	LWSYNC
again:
	LDAR	(R3),R6
	AND	R4, R6, R7
	STDCCC	R7, (R3)
	BNE	again
	MOVD	R6, ret+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-24
	JMP	·And64(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-24
	JMP	·Or64(SB)
