// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// bool Cas(uint32 *ptr, uint32 old, uint32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime∕internal∕atomic·Cas(SB), NOSPLIT, $0-17
	MOVD	ptr+0(FP), R0
	MOVW	old+8(FP), R1
	MOVW	new+12(FP), R2
again:
	LDAXRW	(R0), R3
	CMPW	R1, R3
	BNE	ok
	STLXRW	R2, (R0), R3
	CBNZ	R3, again
ok:
	CSET	EQ, R0
	MOVB	R0, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Casuintptr(SB), NOSPLIT, $0-25
	B	runtime∕internal∕atomic·Cas64(SB)

TEXT runtime∕internal∕atomic·Loaduintptr(SB), NOSPLIT, $-8-16
	B	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Loaduint(SB), NOSPLIT, $-8-16
	B	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Storeuintptr(SB), NOSPLIT, $0-16
	B	runtime∕internal∕atomic·Store64(SB)

TEXT runtime∕internal∕atomic·Xadduintptr(SB), NOSPLIT, $0-24
	B	runtime∕internal∕atomic·Xadd64(SB)

TEXT runtime∕internal∕atomic·Loadint64(SB), NOSPLIT, $0-16
	B	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Xaddint64(SB), NOSPLIT, $0-24
	B	runtime∕internal∕atomic·Xadd64(SB)

// bool Casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime∕internal∕atomic·Casp1(SB), NOSPLIT, $0-25
	B runtime∕internal∕atomic·Cas64(SB)
