// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Store(ptr *uint32, val uint32)
TEXT ·Store(SB), NOSPLIT, $0
	MOVD	ptr+0(FP), R2
	MOVWZ	val+8(FP), R3
	MOVW	R3, 0(R2)
	SYNC
	RET

// func Store8(ptr *uint8, val uint8)
TEXT ·Store8(SB), NOSPLIT, $0
	MOVD	ptr+0(FP), R2
	MOVB	val+8(FP), R3
	MOVB	R3, 0(R2)
	SYNC
	RET

// func Store64(ptr *uint64, val uint64)
TEXT ·Store64(SB), NOSPLIT, $0
	MOVD	ptr+0(FP), R2
	MOVD	val+8(FP), R3
	MOVD	R3, 0(R2)
	SYNC
	RET

// func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer)
TEXT ·StorepNoWB(SB), NOSPLIT, $0
	MOVD	ptr+0(FP), R2
	MOVD	val+8(FP), R3
	MOVD	R3, 0(R2)
	SYNC
	RET

// func Cas(ptr *uint32, old, new uint32) bool
// Atomically:
//	if *ptr == old {
//		*val = new
//		return 1
//	} else {
//		return 0
//	}
TEXT ·Cas(SB), NOSPLIT, $0-17
	MOVD	ptr+0(FP), R3
	MOVWZ	old+8(FP), R4
	MOVWZ	new+12(FP), R5
	CS	R4, R5, 0(R3)    //  if (R4 == 0(R3)) then 0(R3)= R5
	BNE	cas_fail
	MOVB	$1, ret+16(FP)
	RET
cas_fail:
	MOVB	$0, ret+16(FP)
	RET

// func Cas64(ptr *uint64, old, new uint64) bool
// Atomically:
//	if *ptr == old {
//		*ptr = new
//		return 1
//	} else {
//		return 0
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOVD	ptr+0(FP), R3
	MOVD	old+8(FP), R4
	MOVD	new+16(FP), R5
	CSG	R4, R5, 0(R3)    //  if (R4 == 0(R3)) then 0(R3)= R5
	BNE	cas64_fail
	MOVB	$1, ret+24(FP)
	RET
cas64_fail:
	MOVB	$0, ret+24(FP)
	RET

// func Casuintptr(ptr *uintptr, old, new uintptr) bool
TEXT ·Casuintptr(SB), NOSPLIT, $0-25
	BR	·Cas64(SB)

// func CasRel(ptr *uint32, old, new uint32) bool
TEXT ·CasRel(SB), NOSPLIT, $0-17
	BR	·Cas(SB)

// func Loaduintptr(ptr *uintptr) uintptr
TEXT ·Loaduintptr(SB), NOSPLIT, $0-16
	BR	·Load64(SB)

// func Loaduint(ptr *uint) uint
TEXT ·Loaduint(SB), NOSPLIT, $0-16
	BR	·Load64(SB)

// func Storeuintptr(ptr *uintptr, new uintptr)
TEXT ·Storeuintptr(SB), NOSPLIT, $0-16
	BR	·Store64(SB)

// func Loadint64(ptr *int64) int64
TEXT ·Loadint64(SB), NOSPLIT, $0-16
	BR	·Load64(SB)

// func Xadduintptr(ptr *uintptr, delta uintptr) uintptr
TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	BR	·Xadd64(SB)

// func Xaddint64(ptr *int64, delta int64) int64
TEXT ·Xaddint64(SB), NOSPLIT, $0-24
	BR	·Xadd64(SB)

// func Casp1(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool
// Atomically:
//	if *ptr == old {
//		*ptr = new
//		return 1
//	} else {
//		return 0
//	}
TEXT ·Casp1(SB), NOSPLIT, $0-25
	BR ·Cas64(SB)

// func Xadd(ptr *uint32, delta int32) uint32
// Atomically:
//	*ptr += delta
//	return *ptr
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	delta+8(FP), R5
	MOVW	(R4), R3
repeat:
	ADD	R5, R3, R6
	CS	R3, R6, (R4) // if R3==(R4) then (R4)=R6 else R3=(R4)
	BNE	repeat
	MOVW	R6, ret+16(FP)
	RET

// func Xadd64(ptr *uint64, delta int64) uint64
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	delta+8(FP), R5
	MOVD	(R4), R3
repeat:
	ADD	R5, R3, R6
	CSG	R3, R6, (R4) // if R3==(R4) then (R4)=R6 else R3=(R4)
	BNE	repeat
	MOVD	R6, ret+16(FP)
	RET

// func Xchg(ptr *uint32, new uint32) uint32
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	new+8(FP), R3
	MOVW	(R4), R6
repeat:
	CS	R6, R3, (R4) // if R6==(R4) then (R4)=R3 else R6=(R4)
	BNE	repeat
	MOVW	R6, ret+16(FP)
	RET

// func Xchg64(ptr *uint64, new uint64) uint64
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	new+8(FP), R3
	MOVD	(R4), R6
repeat:
	CSG	R6, R3, (R4) // if R6==(R4) then (R4)=R3 else R6=(R4)
	BNE	repeat
	MOVD	R6, ret+16(FP)
	RET

// func Xchguintptr(ptr *uintptr, new uintptr) uintptr
TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	BR	·Xchg64(SB)

// func Or8(addr *uint8, v uint8)
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	// We don't have atomic operations that work on individual bytes so we
	// need to align addr down to a word boundary and create a mask
	// containing v to OR with the entire word atomically.
	MOVD	$(3<<3), R5
	RXSBG	$59, $60, $3, R3, R5 // R5 = 24 - ((addr % 4) * 8) = ((addr & 3) << 3) ^ (3 << 3)
	ANDW	$~3, R3              // R3 = floor(addr, 4) = addr &^ 3
	SLW	R5, R4               // R4 = uint32(v) << R5
	LAO	R4, R6, 0(R3)        // R6 = *R3; *R3 |= R4; (atomic)
	RET

// func And8(addr *uint8, v uint8)
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	// We don't have atomic operations that work on individual bytes so we
	// need to align addr down to a word boundary and create a mask
	// containing v to AND with the entire word atomically.
	ORW	$~0xff, R4           // R4 = uint32(v) | 0xffffff00
	MOVD	$(3<<3), R5
	RXSBG	$59, $60, $3, R3, R5 // R5 = 24 - ((addr % 4) * 8) = ((addr & 3) << 3) ^ (3 << 3)
	ANDW	$~3, R3              // R3 = floor(addr, 4) = addr &^ 3
	RLL	R5, R4, R4           // R4 = rotl(R4, R5)
	LAN	R4, R6, 0(R3)        // R6 = *R3; *R3 &= R4; (atomic)
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LAO	R4, R6, 0(R3)        // R6 = *R3; *R3 |= R4; (atomic)
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	LAN	R4, R6, 0(R3)        // R6 = *R3; *R3 &= R4; (atomic)
	RET
