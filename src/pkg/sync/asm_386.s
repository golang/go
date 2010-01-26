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
TEXT ·cas(SB), 7, $0
	MOVL	4(SP), BX
	MOVL	8(SP), AX
	MOVL	12(SP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ ok
	MOVL	$0, 16(SP)
	RET
ok:
	MOVL	$1, 16(SP)
	RET
