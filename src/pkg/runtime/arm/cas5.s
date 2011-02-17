// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "arm/asm.h"

// This version works on pre v6 architectures

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;

TEXT runtime·cas(SB),7,$0
	MOVW	0(FP), R0	// *val
	MOVW	4(FP), R1	// old
	MOVW	8(FP), R2	// new
	MOVW	$1, R3
	MOVW	$runtime·cas_mutex(SB), R4
	SWPW	(R4), R3	// acquire mutex
	CMP		$0, R3
	BNE		fail0
	
	MOVW	(R0), R5
	CMP		R1, R5
	BNE		fail1
	
	MOVW	R2, (R0)	
	MOVW	R3, (R4)	// release mutex
	MOVW	$1, R0
	RET
fail1:	
	MOVW	R3, (R4)	// release mutex
fail0:
	MOVW	$0, R0
	RET

// bool casp(void **p, void *old, void *new)
// Atomically:
//	if(*p == old){
//		*p = new;
//		return 1;
//	}else
//		return 0;

TEXT runtime·casp(SB),7,$0
	MOVW	0(FP), R0	// *p
	MOVW	4(FP), R1	// old
	MOVW	8(FP), R2	// new
	MOVW	$1, R3
	MOVW	$runtime·cas_mutex(SB), R4
	SWPW	(R4), R3	// acquire mutex
	CMP		$0, R3
	BNE		failp0
	
	MOVW	(R0), R5
	CMP		R1, R5
	BNE		failp1
	
	MOVW	R2, (R0)	
	MOVW	R3, (R4)	// release mutex
	MOVW	$1, R0
	RET
failp1:	
	MOVW	R3, (R4)	// release mutex
failp0:
	MOVW	$0, R0
	RET

DATA runtime·cas_mutex(SB)/4, $0
GLOBL runtime·cas_mutex(SB), $4
