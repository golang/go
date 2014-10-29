// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// NaCl entry on 32-bit x86 has DI pointing at the arg block, which contains:
//
//	0(DI) - cleanup function pointer, always 0
//	4(DI) - envc
//	8(DI) - argc
//	12(DI) - argv, then 0, then envv, then 0, then auxv
// NaCl entry here is almost the same, except that there
// is no saved caller PC, so 0(FP) is -8(FP) and so on. 
TEXT _rt0_amd64p32_nacl(SB),NOSPLIT,$16
	MOVL	DI, 0(SP)
	CALL	runtime·nacl_sysinfo(SB)
	MOVL	0(SP), DI
	MOVL	8(DI), AX
	LEAL	12(DI), BX
	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	CALL	main(SB)
	INT	$3

TEXT main(SB),NOSPLIT,$0
	// Uncomment for fake time like on Go Playground.
	//MOVQ	$1257894000000000000, AX
	//MOVQ	AX, runtime·faketime(SB)
	JMP	runtime·rt0_go(SB)
