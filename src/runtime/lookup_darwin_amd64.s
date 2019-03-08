// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    PUSHQ   BP
    MOVQ    SP, BP
    MOVL    24(DI), R8  // arg 5 anslen
    MOVQ    16(DI), CX  // arg 4 answer
    MOVL    8(DI), SI   // arg 2 class
    MOVQ    12(DI), DX  // arg 3 type
    MOVQ    0(DI), DI   // arg 1 name
    CALL    libc_res_search(SB)
    POPQ    BP
    RET
