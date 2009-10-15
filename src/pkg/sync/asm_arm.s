// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func cas(val *int32, old, new int32) bool
// Atomically:
//	if *val == old {
//		*val = new;
//		return true;
//	}else
//		return false;

TEXT	sync·cas(SB),7,$0
	MOVW	0(FP), R1	// *val
	MOVW	4(FP), R2	// old
	MOVW	8(FP), R3	// new
l:
	LDREX	(R1), R0
	CMP		R0, R2
	BNE		fail
	STREX	R3, (R1), R0
	CMP		$0, R0
	BNE		l
	MOVW	$1, R0
	MOVW	R0, 16(SP)
	RET
fail:
	MOVW	$0, R0
	MOVW	R0, 16(SP)
	RET
