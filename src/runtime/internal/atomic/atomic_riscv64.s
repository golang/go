// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// RISC-V's atomic operations have two bits, aq ("acquire") and rl ("release"),
// which may be toggled on and off. Their precise semantics are defined in
// section 6.3 of the specification, but the basic idea is as follows:
//
//   - If neither aq nor rl is set, the CPU may reorder the atomic arbitrarily.
//     It guarantees only that it will execute atomically.
//
//   - If aq is set, the CPU may move the instruction backward, but not forward.
//
//   - If rl is set, the CPU may move the instruction forward, but not backward.
//
//   - If both are set, the CPU may not reorder the instruction at all.
//
// These four modes correspond to other well-known memory models on other CPUs.
// On ARM, aq corresponds to a dmb ishst, aq+rl corresponds to a dmb ish. On
// Intel, aq corresponds to an lfence, rl to an sfence, and aq+rl to an mfence
// (or a lock prefix).
//
// Go's memory model requires that
//   - if a read happens after a write, the read must observe the write, and
//     that
//   - if a read happens concurrently with a write, the read may observe the
//     write.
// aq is sufficient to guarantee this, so that's what we use here. (This jibes
// with ARM, which uses dmb ishst.)

#include "textflag.h"

// Atomically:
//      if(*val == *old){
//              *val = new;
//              return 1;
//      } else {
//              return 0;
//      }

TEXT ·Cas(SB), NOSPLIT, $0-17
	MOV	ptr+0(FP), A0
	MOVW	old+8(FP), A1
	MOVW	new+12(FP), A2
cas_again:
	LRW	(A0), A3
	BNE	A3, A1, cas_fail
	SCW	A2, (A0), A4
	BNE	A4, ZERO, cas_again
	MOV	$1, A0
	MOVB	A0, ret+16(FP)
	RET
cas_fail:
	MOV	$0, A0
	MOV	A0, ret+16(FP)
	RET

// func Cas64(ptr *uint64, old, new uint64) bool
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOV	ptr+0(FP), A0
	MOV	old+8(FP), A1
	MOV	new+16(FP), A2
cas_again:
	LRD	(A0), A3
	BNE	A3, A1, cas_fail
	SCD	A2, (A0), A4
	BNE	A4, ZERO, cas_again
	MOV	$1, A0
	MOVB	A0, ret+24(FP)
	RET
cas_fail:
	MOVB	ZERO, ret+24(FP)
	RET

// func Load(ptr *uint32) uint32
TEXT ·Load(SB),NOSPLIT|NOFRAME,$0-12
	MOV	ptr+0(FP), A0
	LRW	(A0), A0
	MOVW	A0, ret+8(FP)
	RET

// func Load8(ptr *uint8) uint8
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$0-9
	MOV	ptr+0(FP), A0
	FENCE
	MOVBU	(A0), A1
	FENCE
	MOVB	A1, ret+8(FP)
	RET

// func Load64(ptr *uint64) uint64
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$0-16
	MOV	ptr+0(FP), A0
	LRD	(A0), A0
	MOV	A0, ret+8(FP)
	RET

// func Store(ptr *uint32, val uint32)
TEXT ·Store(SB), NOSPLIT, $0-12
	MOV	ptr+0(FP), A0
	MOVW	val+8(FP), A1
	AMOSWAPW A1, (A0), ZERO
	RET

// func Store8(ptr *uint8, val uint8)
TEXT ·Store8(SB), NOSPLIT, $0-9
	MOV	ptr+0(FP), A0
	MOVBU	val+8(FP), A1
	FENCE
	MOVB	A1, (A0)
	FENCE
	RET

// func Store64(ptr *uint64, val uint64)
TEXT ·Store64(SB), NOSPLIT, $0-16
	MOV	ptr+0(FP), A0
	MOV	val+8(FP), A1
	AMOSWAPD A1, (A0), ZERO
	RET

TEXT ·Casp1(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·Casuintptr(SB),NOSPLIT,$0-25
	JMP	·Cas64(SB)

TEXT ·CasRel(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

TEXT ·Loaduintptr(SB),NOSPLIT,$0-16
	JMP	·Load64(SB)

TEXT ·Storeuintptr(SB),NOSPLIT,$0-16
	JMP	·Store64(SB)

TEXT ·Loaduint(SB),NOSPLIT,$0-16
	JMP ·Loaduintptr(SB)

TEXT ·Loadint64(SB),NOSPLIT,$0-16
	JMP ·Loaduintptr(SB)

TEXT ·Xaddint64(SB),NOSPLIT,$0-24
	MOV	ptr+0(FP), A0
	MOV	delta+8(FP), A1
	AMOADDD A1, (A0), A0
	ADD	A0, A1, A0
	MOVW	A0, ret+16(FP)
	RET

TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-12
	JMP	·Load(SB)

TEXT ·LoadAcq64(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

TEXT ·LoadAcquintptr(SB),NOSPLIT|NOFRAME,$0-16
	JMP	·Load64(SB)

// func Loadp(ptr unsafe.Pointer) unsafe.Pointer
TEXT ·Loadp(SB),NOSPLIT,$0-16
	JMP	·Load64(SB)

// func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer)
TEXT ·StorepNoWB(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreRel(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·StoreRel64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

// func Xchg(ptr *uint32, new uint32) uint32
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOV	ptr+0(FP), A0
	MOVW	new+8(FP), A1
	AMOSWAPW A1, (A0), A1
	MOVW	A1, ret+16(FP)
	RET

// func Xchg64(ptr *uint64, new uint64) uint64
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOV	ptr+0(FP), A0
	MOV	new+8(FP), A1
	AMOSWAPD A1, (A0), A1
	MOV	A1, ret+16(FP)
	RET

// Atomically:
//      *val += delta;
//      return *val;

// func Xadd(ptr *uint32, delta int32) uint32
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOV	ptr+0(FP), A0
	MOVW	delta+8(FP), A1
	AMOADDW A1, (A0), A2
	ADD	A2,A1,A0
	MOVW	A0, ret+16(FP)
	RET

// func Xadd64(ptr *uint64, delta int64) uint64
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOV	ptr+0(FP), A0
	MOV	delta+8(FP), A1
	AMOADDD A1, (A0), A2
	ADD	A2, A1, A0
	MOV	A0, ret+16(FP)
	RET

// func Xadduintptr(ptr *uintptr, delta uintptr) uintptr
TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

// func Xchguintptr(ptr *uintptr, new uintptr) uintptr
TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

// func And8(ptr *uint8, val uint8)
TEXT ·And8(SB), NOSPLIT, $0-9
	MOV	ptr+0(FP), A0
	MOVBU	val+8(FP), A1
	AND	$3, A0, A2
	AND	$-4, A0
	SLL	$3, A2
	XOR	$255, A1
	SLL	A2, A1
	XOR	$-1, A1
	AMOANDW A1, (A0), ZERO
	RET

// func Or8(ptr *uint8, val uint8)
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOV	ptr+0(FP), A0
	MOVBU	val+8(FP), A1
	AND	$3, A0, A2
	AND	$-4, A0
	SLL	$3, A2
	SLL	A2, A1
	AMOORW	A1, (A0), ZERO
	RET

// func And(ptr *uint32, val uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOV	ptr+0(FP), A0
	MOVW	val+8(FP), A1
	AMOANDW	A1, (A0), ZERO
	RET

// func Or(ptr *uint32, val uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOV	ptr+0(FP), A0
	MOVW	val+8(FP), A1
	AMOORW	A1, (A0), ZERO
	RET
