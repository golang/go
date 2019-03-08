// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    SUB    $16, RSP
    MOVW   24(R0), R4 // arg 5 anslen
    MOVD   16(R0), R3 // arg 4 answer
    MOVW   8(R0), R1  // arg 2 class 
    MOVD   12(R0), R2 // arg 3 type
    MOVD   0(R0), R0  // arg 1 name
    BL     libc_res_search(SB)
    RET