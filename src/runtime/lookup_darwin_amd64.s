// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_ninit_trampoline(SB),NOSPLIT,$0
    PUSHQ    BP
    MOVQ     SP, BP
    MOVQ     0(DI), DI  // arg 1 statp
    CALL     libc_res_ninit(SB)
    XORL     DX, DX
    CMPQ     AX, $-1
    JNE ok
    CALL     libc_error(SB)
    MOVLQSX  (AX), DX        // errno
    XORL     AX, AX
ok:
    POPQ    BP
    RET

TEXT runtime·res_nsearch_trampoline(SB),NOSPLIT,$0
    PUSHQ    BP
    MOVQ     SP, BP
    MOVL     32(DI), R9  // arg 6 anslen
    MOVQ     24(DI), R8  // arg 5 answer
    MOVL     12(DI), DX  // arg 3 class
    MOVQ     16(DI), CX  // arg 4 type
    MOVQ     8(DI), SI   // arg 2 name
    MOVQ     0(DI), DI   // arg 1 statp
    CALL     libc_res_nsearch(SB)
    XORL     DX, DX
    CMPQ     AX, $-1
    JNE ok
    CALL     libc_error(SB)
    MOVLQSX  (AX), DX        // errno
    XORL     AX, AX
ok:
    POPQ    BP
    RET
