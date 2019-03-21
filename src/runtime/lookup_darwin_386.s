// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_nsearch_trampoline(SB),NOSPLIT,$0
    PUSHL   BP
    MOVL    SP, BP
    SUBL    $24, SP
    MOVL    32(SP), CX
    MOVL    20(CX), AX      // arg 6 anslen
    MOVL    AX, 20(SP)
    MOVL    16(CX), AX      // arg 5 answer
    MOVL    AX, 16(SP)
    MOVL    8(CX), AX       // arg 3 class
    MOVL    AX, 8(SP)
    MOVL    12(CX), AX       // arg 4 type
    MOVL    AX, 12(SP)
    MOVL    4(CX), AX       // arg 2 name
    MOVL    AX, 4(SP)
    MOVL    4(CX), AX       // arg 2 name
    MOVL    AX, 4(SP)
    MOVL    0(CX), AX       // arg 1 statp
    MOVL    AX, 0(SP)
    CALL    libc_res_nsearch(SB)
    XORL    DX, DX
    CMPL    AX, $-1
    JNE ok
    CALL    libc_error(SB)
    MOVL    (AX), DX        // errno
    XORL    AX, AX
ok:
    MOVL    BP, SP
    POPL    BP
    RET

TEXT runtime·res_ninit_trampoline(SB),NOSPLIT,$0
    PUSHL   BP
    MOVL    SP, BP
    SUBL    $8, SP
    MOVL    16(SP), CX
    MOVL    0(CX), AX  // arg 1 statp
    MOVL    AX, 0(SP)
    CALL    libc_res_ninit(SB)
    XORL    DX, DX
    CMPL    AX, $-1
    JNE ok
    CALL    libc_error(SB)
    MOVL    (AX), DX
    XORL    AX, AX
ok:
    MOVL    BP, SP
    POPL    BP
    RET
