// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_init_trampoline(SB),NOSPLIT,$0
    PUSHQ    BP
    MOVQ     SP, BP
    CALL     libc_res_init(SB)
    CMPQ     AX, $-1
    JNE ok
    CALL     libc_error(SB)
ok:
    POPQ    BP
    RET

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    PUSHQ    BP
    MOVQ     SP, BP
    MOVQ     DI, BX   // move DI into BX to preserve struct addr
    MOVL     24(BX), R8  // arg 5 anslen
    MOVQ     16(BX), CX  // arg 4 answer
    MOVL     12(BX), DX  // arg 3 type
    MOVL     8(BX), SI   // arg 2 class
    MOVQ     0(BX), DI   // arg 1 name
    CALL     libc_res_search(SB)
    XORL     DX, DX
    CMPQ     AX, $-1
    JNE ok
    CALL     libc_error(SB)
    MOVLQSX  (AX), DX             // move return from libc_error into DX
    XORL     AX, AX               // size on error is 0
ok:
    MOVL    AX, 28(BX) // size
    MOVL    DX, 32(BX) // error code
    POPQ    BP
    RET
