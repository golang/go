// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), CX
	MOVQ	CX, BX
	ANDQ	$3, BX
	SHRQ	$2, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSL
	MOVQ	BX, CX
	REP
	STOSB
	// Note: we zero only 4 bytes at a time so that the tail is at most
	// 3 bytes. That guarantees that we aren't zeroing pointers with STOSB.
	// See issue 13160.
	RET
