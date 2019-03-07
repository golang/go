// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    PUSHL   BP
    MOVL    SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
    MOVL    16(CX), AX      // arg 5 anslen
    MOVL    AX, 16(SP)
    MOVL    12(CX), AX      // arg 4 answer
    MOVL    AX, 12(SP)
    MOVL    4(CX), AX       // arg 2 class
    MOVL    AX, 4(SP)
    MOVL    8(CX), AX       // arg 3 type 
    MOVL    AX, 8(SP)       
    MOVL    0(CX), AX       // arg 1 name
    MOVL    AX, 0(SP)
    CALL    libc_res_search(SB)
    MOVL    BP, SP
    POPL    BP
    RET
