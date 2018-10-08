// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	// Instructions that were encoded with BYTE sequences.
	// Included to simplify validation of CL that fixed that.
	MOVQ (AX), M0  // 0f6f00
	MOVQ M0, 8(SP) // 0f7f442408
	MOVQ 8(SP), M0 // 0f6f442408
	MOVQ M0, (AX)  // 0f7f00
	MOVQ M0, (BX)  // 0f7f03
	// On non-64bit arch, Go asm allowed uint32 offsets instead of int32.
	// These tests check that property for backwards-compatibility.
	MOVL 2147483648(AX), AX  // 8b8000000080
	MOVL -2147483648(AX), AX // 8b8000000080
	ADDL 2147483648(AX), AX  // 038000000080
	ADDL -2147483648(AX), AX // 038000000080
	// Make sure MOV CR/DR continues to work after changing its movtabs.
	MOVL CR0, AX // 0f20c0
	MOVL CR0, DX // 0f20c2
	MOVL CR4, DI // 0f20e7
	MOVL AX, CR0 // 0f22c0
	MOVL DX, CR0 // 0f22c2
	MOVL DI, CR4 // 0f22e7
	MOVL DR0, AX // 0f21c0
	MOVL DR6, DX // 0f21f2
	MOVL DR7, SI // 0f21fe
	// Test other movtab entries.
	PUSHL SS // 16
	PUSHL FS // 0fa0
	POPL FS  // 0fa1
	POPL SS  // 17
	// End of tests.
	RET
