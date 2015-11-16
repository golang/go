// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// uint32 runtime∕internal∕atomic·Load(uint32 volatile* addr)
TEXT ·Load(SB),NOSPLIT,$-8-12
	MOVD	ptr+0(FP), R0
	LDARW	(R0), R0
	MOVW	R0, ret+8(FP)
	RET

// uint64 runtime∕internal∕atomic·Load64(uint64 volatile* addr)
TEXT ·Load64(SB),NOSPLIT,$-8-16
	MOVD	ptr+0(FP), R0
	LDAR	(R0), R0
	MOVD	R0, ret+8(FP)
	RET

// void *runtime∕internal∕atomic·Loadp(void *volatile *addr)
TEXT ·Loadp(SB),NOSPLIT,$-8-16
	MOVD	ptr+0(FP), R0
	LDAR	(R0), R0
	MOVD	R0, ret+8(FP)
	RET

TEXT runtime∕internal∕atomic·Storep1(SB), NOSPLIT, $0-16
	B	runtime∕internal∕atomic·Store64(SB)

TEXT runtime∕internal∕atomic·Store(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R0
	MOVW	val+8(FP), R1
	STLRW	R1, (R0)
	RET

TEXT runtime∕internal∕atomic·Store64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R0
	MOVD	val+8(FP), R1
	STLR	R1, (R0)
	RET

TEXT runtime∕internal∕atomic·Xchg(SB), NOSPLIT, $0-20
again:
	MOVD	ptr+0(FP), R0
	MOVW	new+8(FP), R1
	LDAXRW	(R0), R2
	STLXRW	R1, (R0), R3
	CBNZ	R3, again
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xchg64(SB), NOSPLIT, $0-24
again:
	MOVD	ptr+0(FP), R0
	MOVD	new+8(FP), R1
	LDAXR	(R0), R2
	STLXR	R1, (R0), R3
	CBNZ	R3, again
	MOVD	R2, ret+16(FP)
	RET

// bool runtime∕internal∕atomic·Cas64(uint64 *ptr, uint64 old, uint64 new)
// Atomically:
//      if(*val == *old){
//              *val = new;
//              return 1;
//      } else {
//              return 0;
//      }
TEXT runtime∕internal∕atomic·Cas64(SB), NOSPLIT, $0-25
	MOVD	ptr+0(FP), R0
	MOVD	old+8(FP), R1
	MOVD	new+16(FP), R2
again:
	LDAXR	(R0), R3
	CMP	R1, R3
	BNE	ok
	STLXR	R2, (R0), R3
	CBNZ	R3, again
ok:
	CSET	EQ, R0
	MOVB	R0, ret+24(FP)
	RET

// uint32 xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//      *val += delta;
//      return *val;
TEXT runtime∕internal∕atomic·Xadd(SB), NOSPLIT, $0-20
again:
	MOVD	ptr+0(FP), R0
	MOVW	delta+8(FP), R1
	LDAXRW	(R0), R2
	ADDW	R2, R1, R2
	STLXRW	R2, (R0), R3
	CBNZ	R3, again
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xadd64(SB), NOSPLIT, $0-24
again:
	MOVD	ptr+0(FP), R0
	MOVD	delta+8(FP), R1
	LDAXR	(R0), R2
	ADD	R2, R1, R2
	STLXR	R2, (R0), R3
	CBNZ	R3, again
	MOVD	R2, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xchguintptr(SB), NOSPLIT, $0-24
	B	runtime∕internal∕atomic·Xchg64(SB)
