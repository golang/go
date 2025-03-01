// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: some of these functions are semantically inlined
// by the compiler (in src/cmd/compile/internal/gc/ssa.go).

#include "textflag.h"

TEXT ·Loaduintptr(SB), NOSPLIT, $0-16
	JMP	·Load64(SB)

TEXT ·Loaduint(SB), NOSPLIT, $0-16
	JMP	·Load64(SB)

TEXT ·Loadint32(SB), NOSPLIT, $0-12
	JMP	·Load(SB)

TEXT ·Loadint64(SB), NOSPLIT, $0-16
	JMP	·Load64(SB)

// bool Cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
//  }
TEXT ·Cas(SB),NOSPLIT,$0-17
	MOVQ	ptr+0(FP), BX
	MOVL	old+8(FP), AX
	MOVL	new+12(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	SETEQ	ret+16(FP)
	RET

// bool	·Cas64(uint64 *val, uint64 old, uint64 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-25
	MOVQ	ptr+0(FP), BX
	MOVQ	old+8(FP), AX
	MOVQ	new+16(FP), CX
	LOCK
	CMPXCHGQ	CX, 0(BX)
	SETEQ	ret+24(FP)
	RET

// bool Casp1(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
//  }
TEXT ·Casp1(SB), NOSPLIT, $0-25
	MOVQ	ptr+0(FP), BX
	MOVQ	old+8(FP), AX
	MOVQ	new+16(FP), CX
	LOCK
	CMPXCHGQ	CX, 0(BX)
	SETEQ	ret+24(FP)
	RET

TEXT ·Casint32(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

TEXT ·Casint64(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·Casuintptr(SB), NOSPLIT, $0-25
	JMP	·Cas64(SB)

TEXT ·CasRel(SB), NOSPLIT, $0-17
	JMP	·Cas(SB)

// uint32 Xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd(SB), NOSPLIT, $0-20
	MOVQ	ptr+0(FP), BX
	MOVL	delta+8(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	CX, AX
	MOVL	AX, ret+16(FP)
	RET

// uint64 Xadd64(uint64 volatile *val, int64 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT ·Xadd64(SB), NOSPLIT, $0-24
	MOVQ	ptr+0(FP), BX
	MOVQ	delta+8(FP), AX
	MOVQ	AX, CX
	LOCK
	XADDQ	AX, 0(BX)
	ADDQ	CX, AX
	MOVQ	AX, ret+16(FP)
	RET

TEXT ·Xaddint32(SB), NOSPLIT, $0-20
	JMP	·Xadd(SB)

TEXT ·Xaddint64(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

TEXT ·Xadduintptr(SB), NOSPLIT, $0-24
	JMP	·Xadd64(SB)

// uint8 Xchg(ptr *uint8, new uint8)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg8(SB), NOSPLIT, $0-17
	MOVQ	ptr+0(FP), BX
	MOVB	new+8(FP), AX
	XCHGB	AX, 0(BX)
	MOVB	AX, ret+16(FP)
	RET

// uint32 Xchg(ptr *uint32, new uint32)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg(SB), NOSPLIT, $0-20
	MOVQ	ptr+0(FP), BX
	MOVL	new+8(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, ret+16(FP)
	RET

// uint64 Xchg64(ptr *uint64, new uint64)
// Atomically:
//	old := *ptr;
//	*ptr = new;
//	return old;
TEXT ·Xchg64(SB), NOSPLIT, $0-24
	MOVQ	ptr+0(FP), BX
	MOVQ	new+8(FP), AX
	XCHGQ	AX, 0(BX)
	MOVQ	AX, ret+16(FP)
	RET

TEXT ·Xchgint32(SB), NOSPLIT, $0-20
	JMP	·Xchg(SB)

TEXT ·Xchgint64(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

TEXT ·Xchguintptr(SB), NOSPLIT, $0-24
	JMP	·Xchg64(SB)

TEXT ·StorepNoWB(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), BX
	MOVQ	val+8(FP), AX
	XCHGQ	AX, 0(BX)
	RET

TEXT ·Store(SB), NOSPLIT, $0-12
	MOVQ	ptr+0(FP), BX
	MOVL	val+8(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT ·Store8(SB), NOSPLIT, $0-9
	MOVQ	ptr+0(FP), BX
	MOVB	val+8(FP), AX
	XCHGB	AX, 0(BX)
	RET

TEXT ·Store64(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), BX
	MOVQ	val+8(FP), AX
	XCHGQ	AX, 0(BX)
	RET

TEXT ·Storeint32(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·Storeint64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·Storeuintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreRel(SB), NOSPLIT, $0-12
	JMP	·Store(SB)

TEXT ·StoreRel64(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-16
	JMP	·Store64(SB)

// void	·Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-9
	MOVQ	ptr+0(FP), AX
	MOVB	val+8(FP), BX
	LOCK
	ORB	BX, (AX)
	RET

// void	·And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-9
	MOVQ	ptr+0(FP), AX
	MOVB	val+8(FP), BX
	LOCK
	ANDB	BX, (AX)
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-12
	MOVQ	ptr+0(FP), AX
	MOVL	val+8(FP), BX
	LOCK
	ORL	BX, (AX)
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-12
	MOVQ	ptr+0(FP), AX
	MOVL	val+8(FP), BX
	LOCK
	ANDL	BX, (AX)
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-20
	MOVQ	ptr+0(FP), BX
	MOVL	val+8(FP), CX
casloop:
	MOVL 	CX, DX
	MOVL	(BX), AX
	ORL	AX, DX
	LOCK
	CMPXCHGL	DX, (BX)
	JNZ casloop
	MOVL 	AX, ret+16(FP)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-20
	MOVQ	ptr+0(FP), BX
	MOVL	val+8(FP), CX
casloop:
	MOVL 	CX, DX
	MOVL	(BX), AX
	ANDL	AX, DX
	LOCK
	CMPXCHGL	DX, (BX)
	JNZ casloop
	MOVL 	AX, ret+16(FP)
	RET

// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-24
	MOVQ	ptr+0(FP), BX
	MOVQ	val+8(FP), CX
casloop:
	MOVQ 	CX, DX
	MOVQ	(BX), AX
	ORQ	AX, DX
	LOCK
	CMPXCHGQ	DX, (BX)
	JNZ casloop
	MOVQ 	AX, ret+16(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-24
	MOVQ	ptr+0(FP), BX
	MOVQ	val+8(FP), CX
casloop:
	MOVQ 	CX, DX
	MOVQ	(BX), AX
	ANDQ	AX, DX
	LOCK
	CMPXCHGQ	DX, (BX)
	JNZ casloop
	MOVQ 	AX, ret+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-24
	JMP	·And64(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-24
	JMP	·Or64(SB)
