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

TEXT	cas(SB),7,$0
	MOVW	0(FP), R0	// *val
	MOVW	4(FP), R1	// old
	MOVW	8(FP), R2	// new
	MOVW	$1, R3
	MOVW	$cas_mutex(SB), R4
l:
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
 
DATA cas_mutex(SB)/4, $0
GLOBL cas_mutex(SB), $4
